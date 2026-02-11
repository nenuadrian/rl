from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wandb_utils import log_wandb


@dataclass(frozen=True)
class BinaryExample:
    text: str
    label: int  # 0 -> negative, 1 -> positive


@dataclass
class LMGRPOConfig:
    model_name: str
    dtype: str = "bfloat16"
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_steps: int = 1000
    prompts_per_step: int = 32
    ppo_epochs: int = 4
    minibatch_size: int = 32
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    ent_coef: float = 1e-3
    head_only_steps: int = 0
    eval_every: int = 25
    eval_examples: int = 256
    save_every: int = 200
    reward_std_eps: float = 1e-6
    advantage_mode: str = "ema_baseline"
    normalize_advantages: bool = True
    baseline_momentum: float = 0.9
    kl_coef: float = 0.02
    kl_coef_min: float = 1e-3
    target_ref_kl: float = 0.08
    kl_adaptation_factor: float = 1.5
    kl_coef_up_mult: float = 1.02
    kl_coef_down_div: float = 1.02
    seed: int = 0
    dataset_name: str = "glue"
    dataset_config: str | None = "sst2"
    train_split: str = "train"
    eval_split: str = "validation"
    text_key: str = "sentence"
    label_key: str = "label"
    negative_label_ids: tuple[int, ...] = (0,)
    positive_label_ids: tuple[int, ...] = (1,)
    prompt_template: str = "Sentence: {text}\nSentiment:"
    max_prompt_length: int = 256
    train_subset_size: int | None = None
    eval_subset_size: int | None = 4096
    train_transformer: bool = True


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_name == "bfloat16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device.type == "cuda" else torch.float32
    raise ValueError(f"Unsupported LM dtype: {dtype_name}")


class LMTrainer:
    """Single-GPU PPO trainer with an explicit 2-logit classification head."""

    ACTION_TEXT = {0: "negative", 1: "positive"}

    def __init__(
        self,
        config: LMGRPOConfig,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        seed = self.config.seed
        self.train_rng = random.Random(seed)
        self.model: Any | None = None
        self.reference_model: Any | None = None
        self.policy_head: nn.Linear | None = None
        self.reference_policy_head: nn.Linear | None = None
        self.tokenizer: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.trainable_params: list[nn.Parameter] = []
        self.kl_coef: float = self.config.kl_coef
        self.reward_baseline: float | None = None
        self.train_examples: list[BinaryExample] = []
        self.eval_examples_pool: list[BinaryExample] = []
        self.transformer_trainable: bool = False

        if self.config.head_only_steps < 0:
            raise ValueError(
                f"head_only_steps must be non-negative, got {self.config.head_only_steps}."
            )

    def _ensure_initialized(self) -> None:
        if (
            self.model is not None
            and self.reference_model is not None
            and self.policy_head is not None
            and self.reference_policy_head is not None
            and self.tokenizer is not None
            and self.optimizer is not None
            and self.train_examples
            and self.eval_examples_pool
        ):
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for LM classification training. Install with `pip install datasets`."
            ) from exc

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for LM training. Install with `pip install transformers`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer must define either a pad token or EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token

        model_dtype = _resolve_dtype(self.config.dtype, self.device)
        model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=model_dtype,
        )
        reference_model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=model_dtype,
        )
        model.to(self.device)
        reference_model.to(self.device)
        model.train()
        reference_model.eval()
        reference_model.requires_grad_(False)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(reference_model.config, "use_cache"):
            reference_model.config.use_cache = False

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError(
                f"Unable to infer hidden size from model config for {self.config.model_name}."
            )

        policy_head = nn.Linear(int(hidden_size), 2, bias=True).to(self.device)
        reference_policy_head = nn.Linear(int(hidden_size), 2, bias=True).to(self.device)
        reference_policy_head.load_state_dict(policy_head.state_dict())
        reference_policy_head.eval()
        reference_policy_head.requires_grad_(False)

        self.transformer_trainable = bool(
            self.config.train_transformer and self.config.head_only_steps <= 0
        )
        model.requires_grad_(self.transformer_trainable)

        dataset = load_dataset(
            path=self.config.dataset_name,
            name=self.config.dataset_config,
        )
        train_split = dataset[self.config.train_split]
        eval_split = dataset[self.config.eval_split]

        train_examples = self._build_examples_from_rows(list(train_split))
        eval_examples_pool = self._build_examples_from_rows(list(eval_split))
        train_examples = self._maybe_subsample(
            train_examples,
            self.config.train_subset_size,
            seed=self.config.seed,
        )
        eval_examples_pool = self._maybe_subsample(
            eval_examples_pool,
            self.config.eval_subset_size,
            seed=self.config.seed + 1,
        )
        if not train_examples:
            raise ValueError("No valid binary training examples were found.")
        if not eval_examples_pool:
            raise ValueError("No valid binary evaluation examples were found.")

        self.model = model
        self.reference_model = reference_model
        self.policy_head = policy_head
        self.reference_policy_head = reference_policy_head
        self.tokenizer = tokenizer
        self.optimizer, self.trainable_params = self._build_optimizer(
            include_transformer=self.transformer_trainable
        )
        self.train_examples = train_examples
        self.eval_examples_pool = eval_examples_pool

    def _build_optimizer(
        self,
        include_transformer: bool,
        previous_optimizer: torch.optim.Optimizer | None = None,
        preserve_shared_state: bool = False,
    ) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
        assert self.policy_head is not None
        assert self.model is not None

        trainable_params: list[nn.Parameter] = list(self.policy_head.parameters())
        if include_transformer:
            trainable_params.extend(self.model.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if preserve_shared_state and previous_optimizer is not None:
            old_state = previous_optimizer.state
            for param in trainable_params:
                if param in old_state:
                    optimizer.state[param] = copy.deepcopy(old_state[param])
        return optimizer, trainable_params

    def _set_transformer_trainable(self, enabled: bool, step: int) -> None:
        assert self.model is not None
        assert self.optimizer is not None
        if self.transformer_trainable == enabled:
            return
        old_optimizer = self.optimizer
        self.model.requires_grad_(enabled)
        self.optimizer, self.trainable_params = self._build_optimizer(
            include_transformer=enabled,
            previous_optimizer=old_optimizer,
            preserve_shared_state=True,
        )
        self.transformer_trainable = enabled
        phase = "head+transformer" if enabled else "head-only"
        print(
            f"step={step} switched LM PPO training phase to {phase} "
            "(optimizer rebuilt; new params start with fresh state)"
        )

    def _build_examples_from_rows(self, rows: list[dict[str, Any]]) -> list[BinaryExample]:
        examples: list[BinaryExample] = []
        for row in rows:
            text_raw = row.get(self.config.text_key)
            label_raw = row.get(self.config.label_key)
            if text_raw is None or label_raw is None:
                continue
            text = str(text_raw).strip()
            if not text:
                continue
            label = self._map_label_to_binary(int(label_raw))
            if label is None:
                continue
            examples.append(BinaryExample(text=text, label=label))
        return examples

    @staticmethod
    def _maybe_subsample(
        examples: list[BinaryExample],
        subset_size: int | None,
        seed: int,
    ) -> list[BinaryExample]:
        if subset_size is None or subset_size <= 0 or subset_size >= len(examples):
            return examples
        rng = random.Random(seed)
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        return [examples[i] for i in indices[:subset_size]]

    def _map_label_to_binary(self, raw_label: int) -> int | None:
        negative = set(int(v) for v in self.config.negative_label_ids)
        positive = set(int(v) for v in self.config.positive_label_ids)
        if not negative or not positive:
            raise ValueError("Both negative_label_ids and positive_label_ids must be set.")
        if negative & positive:
            raise ValueError("negative_label_ids and positive_label_ids must be disjoint.")
        if raw_label in negative:
            return 0
        if raw_label in positive:
            return 1
        return None

    def _build_prompt(self, example: BinaryExample) -> str:
        return self.config.prompt_template.format(text=example.text)

    def _sample_train_examples(self) -> list[BinaryExample]:
        batch_size = self.config.prompts_per_step
        if len(self.train_examples) >= batch_size:
            return self.train_rng.sample(self.train_examples, batch_size)
        return [self.train_rng.choice(self.train_examples) for _ in range(batch_size)]

    def _build_eval_examples(self, step: int) -> list[BinaryExample]:
        rng = random.Random(self.config.seed + (step * 104729))
        total = min(self.config.eval_examples, len(self.eval_examples_pool))
        if total <= 0:
            raise ValueError("eval_examples must be > 0")
        indices = list(range(len(self.eval_examples_pool)))
        rng.shuffle(indices)
        return [self.eval_examples_pool[i] for i in indices[:total]]

    def _tokenize_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    @staticmethod
    def _last_token_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device, dtype=torch.long)
        return hidden[batch_idx, last_idx]

    def _policy_logits(
        self,
        model: Any,
        head: nn.Linear,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        state = self._last_token_hidden(hidden=hidden, attention_mask=attention_mask)
        return head(state.float())

    @staticmethod
    def _categorical_kl(
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor:
        policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
        reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
        policy_probs = policy_log_probs.exp()
        return (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)

    def _collect_rollout(
        self,
        examples: list[BinaryExample],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        assert self.model is not None
        assert self.reference_model is not None
        assert self.policy_head is not None
        assert self.reference_policy_head is not None

        prompts = [self._build_prompt(example) for example in examples]
        encoded = self._tokenize_prompts(prompts)
        labels = torch.tensor(
            [example.label for example in examples],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            old_logits = self._policy_logits(
                self.model,
                self.policy_head,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            dist = torch.distributions.Categorical(logits=old_logits)
            actions = dist.sample()
            old_log_probs = dist.log_prob(actions).detach()
            reference_logits = self._policy_logits(
                self.reference_model,
                self.reference_policy_head,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).detach()

        rewards = (actions == labels).to(dtype=torch.float32)
        advantages = self._compute_advantages(rewards=rewards).detach()
        sampled_texts = [self.ACTION_TEXT[int(a.item())] for a in actions]

        rollout = {
            "input_ids": encoded["input_ids"].detach(),
            "attention_mask": encoded["attention_mask"].detach(),
            "labels": labels.detach(),
            "actions": actions.detach(),
            "old_log_probs": old_log_probs.detach(),
            "reference_logits": reference_logits,
            "advantages": advantages,
            "rewards": rewards.detach(),
        }
        return rollout, sampled_texts

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        mode = self.config.advantage_mode
        if mode == "batch_norm":
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            return (rewards - mean) / (std + self.config.reward_std_eps)

        if mode == "ema_baseline":
            batch_mean = float(rewards.mean().item())
            if self.reward_baseline is None:
                self.reward_baseline = batch_mean
            else:
                m = self.config.baseline_momentum
                self.reward_baseline = (m * self.reward_baseline) + (
                    (1.0 - m) * batch_mean
                )
            advantages = rewards - float(self.reward_baseline)
            if self.config.normalize_advantages:
                std = advantages.std(unbiased=False)
                advantages = advantages / (std + self.config.reward_std_eps)
            return advantages

        if mode == "raw":
            return rewards

        raise ValueError(
            f"Unsupported advantage_mode: {mode}. "
            "Use one of: raw, batch_norm, ema_baseline."
        )

    def _optimize(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        assert self.model is not None
        assert self.policy_head is not None
        assert self.optimizer is not None

        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]
        labels = rollout["labels"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]
        reference_logits = rollout["reference_logits"]
        advantages = rollout["advantages"]
        rewards = rollout["rewards"]

        total = int(input_ids.size(0))
        minibatch_size = min(self.config.minibatch_size, total)
        clip_eps = self.config.clip_epsilon

        losses: list[torch.Tensor] = []
        ratio_means: list[torch.Tensor] = []
        approx_kls: list[torch.Tensor] = []
        ref_kls: list[torch.Tensor] = []
        grad_norms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        train_acc_terms: list[torch.Tensor] = []

        self.model.train()
        self.policy_head.train()

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                logits = self._policy_logits(
                    self.model,
                    self.policy_head,
                    input_ids=input_ids[idx],
                    attention_mask=attention_mask[idx],
                )
                log_probs = F.log_softmax(logits.float(), dim=-1)
                action_log_probs = log_probs.gather(
                    -1, actions[idx].unsqueeze(-1)
                ).squeeze(-1)
                log_ratio = action_log_probs - old_log_probs[idx]
                ratios = torch.exp(log_ratio)
                clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_objective = torch.min(
                    ratios * advantages[idx],
                    clipped * advantages[idx],
                )
                ref_kl = self._categorical_kl(
                    policy_logits=logits,
                    reference_logits=reference_logits[idx],
                )
                entropy_per_sample = torch.distributions.Categorical(
                    logits=logits
                ).entropy()
                objective = (
                    policy_objective
                    - (self.kl_coef * ref_kl)
                    + (self.config.ent_coef * entropy_per_sample)
                )
                loss = -objective.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_params,
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = 0.5 * log_ratio.pow(2).mean()
                entropy = entropy_per_sample.mean()
                train_acc = logits.argmax(dim=-1).eq(labels[idx]).float().mean()

                losses.append(loss.detach())
                ratio_means.append(ratios.mean().detach())
                approx_kls.append(approx_kl.detach())
                ref_kls.append(ref_kl.mean().detach())
                grad_norms.append(
                    torch.as_tensor(grad_norm, dtype=torch.float32, device=self.device)
                )
                entropy_terms.append(entropy.detach())
                train_acc_terms.append(train_acc.detach())

        mean_loss = torch.stack(losses).mean().item()
        mean_ratio = torch.stack(ratio_means).mean().item()
        mean_kl = torch.stack(approx_kls).mean().item()
        mean_ref_kl = torch.stack(ref_kls).mean().item()
        mean_grad_norm = torch.stack(grad_norms).mean().item()
        mean_entropy = torch.stack(entropy_terms).mean().item()
        mean_train_acc = torch.stack(train_acc_terms).mean().item()
        reward_mean = rewards.mean().item()
        reward_max = rewards.max().item()
        reward_min = rewards.min().item()
        advantage_std = advantages.std(unbiased=False).item()

        if mean_ref_kl > (self.config.target_ref_kl * self.config.kl_adaptation_factor):
            self.kl_coef *= self.config.kl_coef_up_mult
        elif mean_ref_kl < (
            self.config.target_ref_kl / self.config.kl_adaptation_factor
        ):
            self.kl_coef /= self.config.kl_coef_down_div
        self.kl_coef = float(max(self.kl_coef, self.config.kl_coef_min))

        return {
            "train/loss": mean_loss,
            "train/ratio_mean": mean_ratio,
            "train/approx_kl": mean_kl,
            "train/ref_kl": mean_ref_kl,
            "train/kl_coef": self.kl_coef,
            "train/ent_coef": self.config.ent_coef,
            "train/transformer_trainable": (
                1.0 if self.transformer_trainable else 0.0
            ),
            "train/grad_norm": mean_grad_norm,
            "train/entropy": mean_entropy,
            "train/accuracy": mean_train_acc,
            "train/reward_mean": reward_mean,
            "train/reward_max": reward_max,
            "train/reward_min": reward_min,
            "train/adv_std": advantage_std,
            "train/reward_baseline": (
                float(self.reward_baseline)
                if self.reward_baseline is not None
                else float("nan")
            ),
        }

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, float]:
        assert self.model is not None
        assert self.policy_head is not None

        self.model.eval()
        self.policy_head.eval()

        examples = self._build_eval_examples(step)
        prompts = [self._build_prompt(example) for example in examples]
        labels = torch.tensor(
            [example.label for example in examples],
            dtype=torch.long,
            device=self.device,
        )
        encoded = self._tokenize_prompts(prompts)
        logits = self._policy_logits(
            self.model,
            self.policy_head,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        probs = F.softmax(logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)

        accuracy = preds.eq(labels).float().mean().item()
        positive_rate = preds.float().mean().item()
        mean_confidence = probs.max(dim=-1).values.mean().item()

        self.model.train()
        self.policy_head.train()
        return {
            "eval/accuracy": accuracy,
            "eval/positive_rate": positive_rate,
            "eval/confidence_mean": mean_confidence,
            "eval/return_max": accuracy,
        }

    def _save_checkpoint(self, out_dir: Path, step: int) -> None:
        assert self.model is not None
        assert self.policy_head is not None
        assert self.tokenizer is not None
        checkpoint_dir = out_dir / f"lm_step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        encoder_dir = checkpoint_dir / "encoder"
        tokenizer_dir = checkpoint_dir / "tokenizer"
        self.model.save_pretrained(encoder_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
        torch.save(
            {
                "policy_head_state_dict": self.policy_head.state_dict(),
                "action_text": dict(self.ACTION_TEXT),
                "prompt_template": self.config.prompt_template,
            },
            checkpoint_dir / "policy_head.pt",
        )

    def train(self, out_dir: str) -> None:
        self._ensure_initialized()

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, self.config.num_steps + 1):
            should_train_transformer = bool(
                self.config.train_transformer and step > self.config.head_only_steps
            )
            self._set_transformer_trainable(
                enabled=should_train_transformer,
                step=step,
            )
            examples = self._sample_train_examples()
            rollout, sampled_actions = self._collect_rollout(examples)
            metrics = self._optimize(rollout)

            if step == 1 or step % self.config.eval_every == 0:
                metrics.update(self.evaluate(step=step))

            log_wandb(metrics, step=step, silent=True)

            if step == 1 or step % 10 == 0:
                sample_text = examples[0].text.replace("\n", " ").strip()
                eval_acc = metrics.get("eval/accuracy")
                acc_display = (
                    f"{float(eval_acc):.3f}" if eval_acc is not None else "n/a"
                )
                print(
                    f"step={step} reward={metrics['train/reward_mean']:.3f} "
                    f"acc={acc_display} "
                    f"sample='Sentence: {sample_text[:80]}...' "
                    f"action='{sampled_actions[0]}'"
                )

            if step % self.config.save_every == 0 or step == self.config.num_steps:
                self._save_checkpoint(out_dir=output_dir, step=step)
