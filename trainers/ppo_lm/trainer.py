from __future__ import annotations

import re
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wandb_utils import log_wandb


ANSWER_RE = re.compile(r"#### (\-?[0-9\.,]+)")


def extract_answer(completion: str) -> str | None:
    match = ANSWER_RE.search(completion)
    if not match:
        return None
    answer = match.group(1).strip().replace(",", "")
    return answer


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


def _disable_dropout(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = False
) -> torch.Tensor:
    mean = masked_mean(values, mask)
    centered = (values - mean) * mask
    denom = mask.sum().clamp(min=1.0)
    variance = centered.pow(2).sum() / denom
    if unbiased:
        count = mask.sum()
        if count > 1:
            variance = variance * (count / (count - 1.0))
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(values, mask)
    variance = masked_var(values, mask, unbiased=True)
    return (values - mean) * torch.rsqrt(variance + 1e-8)


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    full_answer: str
    final_answer: str


@dataclass
class LMGRPOConfig:
    model_name: str
    tokenizer_name: str | None = None
    dtype: str = "bfloat16"

    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    adam_eps: float = 1e-8

    num_steps: int = 1000
    prompts_per_step: int = 16
    ppo_epochs: int = 4
    minibatch_size: int = 8

    clip_epsilon: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    ent_coef: float = 5e-4
    max_grad_norm: float = 1.0

    gamma: float = 1.0
    lam: float = 0.95
    whiten_advantages: bool = True
    whiten_rewards: bool = False

    kl_coef: float = 0.02
    adaptive_kl: bool = True
    target_ref_kl: float = 0.08
    kl_horizon: int = 10_000

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    max_prompt_length: int = 384
    max_new_tokens: int = 160
    eval_max_new_tokens: int = 192

    eval_every: int = 25
    eval_examples: int = 128
    save_every: int = 200

    dataset_name: str = "openai/gsm8k"
    dataset_config: str | None = "main"
    train_split: str = "train"
    eval_split: str = "test"
    train_subset_size: int | None = None
    eval_subset_size: int | None = 512

    prompt_template: str = (
        "Solve the following grade-school math problem. "
        "Show your work, then end with `#### <number>`.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    reward_correct: float = 1.0
    reward_has_answer_tag: float = 0.1
    reward_parseable: float = 0.1
    reward_wrong: float = 0.0

    reference_on_cpu: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.prompts_per_step <= 0:
            raise ValueError("prompts_per_step must be > 0")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be > 0")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.eval_max_new_tokens <= 0:
            raise ValueError("eval_max_new_tokens must be > 0")


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = float(init_kl_coef)
        self.target = float(target)
        self.horizon = int(horizon)

    def update(self, current: float, n_steps: int) -> None:
        if self.target <= 0 or self.horizon <= 0:
            return
        proportional_error = float(max(min((current / self.target) - 1.0, 0.2), -0.2))
        mult = 1.0 + (proportional_error * n_steps / self.horizon)
        self.value *= mult


class FixedKLController:
    def __init__(self, kl_coef: float):
        self.value = float(kl_coef)

    def update(self, current: float, n_steps: int) -> None:
        return


class PPOLMTrainer:
    """Single-process PPO trainer for causal LMs token-level PPO flow."""

    def __init__(
        self,
        config: LMGRPOConfig,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.train_rng = random.Random(self.config.seed)

        self.model: Any | None = None
        self.reference_model: Any | None = None
        self.value_head: nn.Linear | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.tokenizer: Any | None = None

        self.train_examples: list[GSM8KExample] = []
        self.eval_examples_pool: list[GSM8KExample] = []

        if self.config.adaptive_kl:
            self.kl_controller: AdaptiveKLController | FixedKLController = (
                AdaptiveKLController(
                    init_kl_coef=self.config.kl_coef,
                    target=self.config.target_ref_kl,
                    horizon=self.config.kl_horizon,
                )
            )
        else:
            self.kl_controller = FixedKLController(self.config.kl_coef)

    @property
    def kl_coef(self) -> float:
        return float(self.kl_controller.value)

    def _ensure_initialized(self) -> None:
        if (
            self.model is not None
            and self.reference_model is not None
            and self.value_head is not None
            and self.optimizer is not None
            and self.tokenizer is not None
            and self.train_examples
            and self.eval_examples_pool
        ):
            return

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for PPO LM training. Install with `pip install datasets`."
            ) from exc

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for PPO LM training. Install with `pip install transformers`."
            ) from exc

        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer must define a pad token or EOS token for generation."
                )
            tokenizer.pad_token = tokenizer.eos_token

        actor_dtype = _resolve_dtype(self.config.dtype, self.device)
        ref_device = (
            torch.device("cpu") if self.config.reference_on_cpu else self.device
        )
        ref_dtype = torch.float32 if ref_device.type == "cpu" else actor_dtype

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=actor_dtype,
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=ref_dtype,
        )

        model.to(self.device)
        reference_model.to(ref_device)
        model.train()
        reference_model.eval()
        reference_model.requires_grad_(False)

        _disable_dropout(model)
        _disable_dropout(reference_model)

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(reference_model.config, "use_cache"):
            reference_model.config.use_cache = False

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden size from model config for {self.config.model_name}."
            )

        value_head = nn.Linear(int(hidden_size), 1, bias=True).to(self.device)

        params: list[nn.Parameter] = list(model.parameters()) + list(
            value_head.parameters()
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )

        dataset = load_dataset(
            path=self.config.dataset_name,
            name=self.config.dataset_config,
        )
        train_rows = list(dataset[self.config.train_split])
        eval_rows = list(dataset[self.config.eval_split])

        train_examples = self._build_examples_from_rows(train_rows)
        eval_examples = self._build_examples_from_rows(eval_rows)

        train_examples = self._maybe_subsample(
            train_examples,
            subset_size=self.config.train_subset_size,
            seed=self.config.seed,
        )
        eval_examples = self._maybe_subsample(
            eval_examples,
            subset_size=self.config.eval_subset_size,
            seed=self.config.seed + 1,
        )

        if not train_examples:
            raise ValueError("No valid training examples were built from the dataset.")
        if not eval_examples:
            raise ValueError("No valid eval examples were built from the dataset.")

        self.model = model
        self.reference_model = reference_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.eval_examples_pool = eval_examples

    def _build_examples_from_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[GSM8KExample]:
        examples: list[GSM8KExample] = []
        for row in rows:
            question_raw = row.get("question")
            answer_raw = row.get("answer")
            if question_raw is None or answer_raw is None:
                continue
            question = str(question_raw).strip()
            full_answer = str(answer_raw)
            final_answer = extract_answer(full_answer)
            if not question or final_answer is None:
                continue
            examples.append(
                GSM8KExample(
                    question=question,
                    full_answer=full_answer,
                    final_answer=final_answer,
                )
            )
        return examples

    @staticmethod
    def _maybe_subsample(
        examples: list[GSM8KExample],
        subset_size: int | None,
        seed: int,
    ) -> list[GSM8KExample]:
        if subset_size is None or subset_size <= 0 or subset_size >= len(examples):
            return examples
        rng = random.Random(seed)
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        return [examples[i] for i in indices[:subset_size]]

    def _build_prompt(self, example: GSM8KExample) -> str:
        return self.config.prompt_template.format(question=example.question)

    def _sample_train_examples(self) -> list[GSM8KExample]:
        batch_size = self.config.prompts_per_step
        if len(self.train_examples) >= batch_size:
            return self.train_rng.sample(self.train_examples, batch_size)
        return [self.train_rng.choice(self.train_examples) for _ in range(batch_size)]

    def _build_eval_examples(self, step: int) -> list[GSM8KExample]:
        rng = random.Random(self.config.seed + (step * 104_729))
        total = min(self.config.eval_examples, len(self.eval_examples_pool))
        if total <= 0:
            raise ValueError("eval_examples must be > 0")
        indices = list(range(len(self.eval_examples_pool)))
        rng.shuffle(indices)
        return [self.eval_examples_pool[i] for i in indices[:total]]

    def _trim_response_tokens(self, generated: list[int]) -> list[int]:
        assert self.tokenizer is not None
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        trimmed: list[int] = []
        for tok in generated:
            if pad_id is not None and tok == pad_id:
                break
            trimmed.append(tok)
            if eos_id is not None and tok == eos_id:
                break

        if trimmed:
            return trimmed

        fallback = eos_id if eos_id is not None else pad_id
        if fallback is None:
            return []
        return [fallback]

    def _generate_responses(
        self,
        prompts: list[str],
        *,
        do_sample: bool,
        max_new_tokens: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        assert self.model is not None
        assert self.tokenizer is not None

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.top_k > 0:
            generation_kwargs["top_k"] = int(self.config.top_k)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        prompt_width = int(input_ids.size(1))
        query_token_ids: list[list[int]] = []
        response_token_ids: list[list[int]] = []
        response_texts: list[str] = []

        for i in range(len(prompts)):
            query_ids = input_ids[i][attention_mask[i].bool()].tolist()
            raw_response = generated[i, prompt_width:].tolist()
            trimmed_response = self._trim_response_tokens(raw_response)

            query_token_ids.append(query_ids)
            response_token_ids.append(trimmed_response)
            response_texts.append(
                self.tokenizer.decode(trimmed_response, skip_special_tokens=True)
            )

        return query_token_ids, response_token_ids, response_texts

    def _score_response(
        self,
        example: GSM8KExample,
        response_text: str,
    ) -> tuple[float, bool, bool, str | None]:
        pred_answer = extract_answer(response_text)
        has_tag = "####" in response_text
        is_correct = pred_answer is not None and pred_answer == example.final_answer

        reward = self.config.reward_wrong
        if has_tag:
            reward += self.config.reward_has_answer_tag
        if pred_answer is not None:
            reward += self.config.reward_parseable
        if is_correct:
            reward += self.config.reward_correct

        return float(reward), has_tag, is_correct, pred_answer

    def _build_training_tensors(
        self,
        episodes: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        pad_id = int(self.tokenizer.pad_token_id)

        max_seq_len = max(
            len(ep["query_token_ids"]) + len(ep["response_token_ids"])
            for ep in episodes
        )

        batch_size = len(episodes)
        input_ids = torch.full(
            (batch_size, max_seq_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.full(
            (batch_size, max_seq_len),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.long,
            device=self.device,
        )

        scores = torch.tensor(
            [float(ep["score"]) for ep in episodes],
            dtype=torch.float32,
            device=self.device,
        )

        for i, ep in enumerate(episodes):
            query_ids = ep["query_token_ids"]
            response_ids = ep["response_token_ids"]
            sequence = query_ids + response_ids
            q_len = len(query_ids)
            seq_len = len(sequence)

            input_ids[i, :seq_len] = torch.tensor(
                sequence, dtype=torch.long, device=self.device
            )
            labels[i, q_len:seq_len] = torch.tensor(
                response_ids, dtype=torch.long, device=self.device
            )
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "scores": scores,
        }

    @staticmethod
    def _token_logprobs_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        action_mask = (shift_labels != -100).to(dtype=shift_logits.dtype)
        safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            dim=2, index=safe_labels.unsqueeze(2)
        ).squeeze(2)
        token_log_probs = token_log_probs * action_mask
        return token_log_probs, action_mask

    @staticmethod
    def _token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    def _compute_reference_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        assert self.reference_model is not None

        ref_device = next(self.reference_model.parameters()).device
        input_ids_ref = input_ids.to(ref_device)
        attention_ref = attention_mask.to(ref_device)
        labels_ref = labels.to(ref_device)

        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=input_ids_ref,
                attention_mask=attention_ref,
                return_dict=True,
                use_cache=False,
            )
            ref_logprobs, _ = self._token_logprobs_from_logits(
                ref_outputs.logits, labels_ref
            )

        return ref_logprobs.to(self.device)

    def _compute_rewards(
        self,
        scores: torch.Tensor,
        shifted_actor_logprobs: torch.Tensor,
        shifted_ref_logprobs: torch.Tensor | None,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if shifted_ref_logprobs is not None:
            kls = shifted_actor_logprobs - shifted_ref_logprobs
            non_score_rewards = -self.kl_coef * kls
        else:
            kls = None
            non_score_rewards = torch.zeros_like(shifted_actor_logprobs)

        rewards = non_score_rewards.clone()

        last_non_pad = attention_mask.sum(dim=1).long() - 1
        last_action_idx = (last_non_pad - 1).clamp(min=0, max=rewards.size(1) - 1)
        row_idx = torch.arange(rewards.size(0), device=rewards.device)
        rewards[row_idx, last_action_idx] += scores

        rewards = rewards * action_mask
        non_score_rewards = non_score_rewards * action_mask
        if kls is not None:
            kls = kls * action_mask

        return (
            rewards.detach(),
            non_score_rewards.detach(),
            (None if kls is None else kls.detach()),
        )

    def _compute_advantages(
        self,
        valid_values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards * action_mask
        valid_values = valid_values * action_mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, action_mask)

        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(
            rewards.size(0), dtype=rewards.dtype, device=rewards.device
        )
        actions_seq_len = rewards.size(1)

        for t in reversed(range(actions_seq_len)):
            next_values = valid_values[:, t + 1] if t < actions_seq_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * next_values - valid_values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            lastgaelam = lastgaelam * action_mask[:, t]
            advantages[:, t] = lastgaelam

        returns = advantages + valid_values
        if self.config.whiten_advantages:
            advantages = masked_whiten(advantages, action_mask)

        return advantages.detach(), returns.detach()

    def _collect_rollout(self, examples: list[GSM8KExample]) -> dict[str, Any]:
        assert self.model is not None
        assert self.value_head is not None

        prompts = [self._build_prompt(ex) for ex in examples]

        self.model.eval()
        query_token_ids, response_token_ids, responses = self._generate_responses(
            prompts,
            do_sample=True,
            max_new_tokens=self.config.max_new_tokens,
        )

        episodes: list[dict[str, Any]] = []
        has_tag_flags: list[float] = []
        correct_flags: list[float] = []
        parseable_flags: list[float] = []

        for i, example in enumerate(examples):
            score, has_tag, is_correct, pred_answer = self._score_response(
                example,
                responses[i],
            )
            episodes.append(
                {
                    "query_token_ids": query_token_ids[i],
                    "response_token_ids": response_token_ids[i],
                    "score": score,
                    "response": responses[i],
                    "target": example.final_answer,
                    "prediction": pred_answer,
                }
            )
            has_tag_flags.append(float(has_tag))
            correct_flags.append(float(is_correct))
            parseable_flags.append(float(pred_answer is not None))

        tensors = self._build_training_tensors(episodes)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
            old_logprobs, action_mask = self._token_logprobs_from_logits(
                outputs.logits,
                tensors["labels"],
            )
            values = self.value_head(outputs.hidden_states[-1].float()).squeeze(-1)

            if self.reference_model is not None:
                ref_logprobs = self._compute_reference_logprobs(
                    tensors["input_ids"],
                    tensors["attention_mask"],
                    tensors["labels"],
                )
            else:
                ref_logprobs = None

            rewards, non_score_rewards, kls = self._compute_rewards(
                tensors["scores"],
                old_logprobs,
                ref_logprobs,
                tensors["attention_mask"],
                action_mask,
            )
            old_valid_values = values[:, :-1] * action_mask
            advantages, returns = self._compute_advantages(
                old_valid_values,
                rewards,
                action_mask,
            )

        self.model.train()

        return {
            "input_ids": tensors["input_ids"],
            "attention_mask": tensors["attention_mask"],
            "labels": tensors["labels"],
            "scores": tensors["scores"],
            "old_logprobs": old_logprobs.detach(),
            "ref_logprobs": None if ref_logprobs is None else ref_logprobs.detach(),
            "old_values": old_valid_values.detach(),
            "returns": returns.detach(),
            "advantages": advantages.detach(),
            "action_mask": action_mask.detach(),
            "rewards": rewards.detach(),
            "non_score_rewards": non_score_rewards.detach(),
            "kls": None if kls is None else kls.detach(),
            "has_tag": torch.tensor(
                has_tag_flags, dtype=torch.float32, device=self.device
            ),
            "is_correct": torch.tensor(
                correct_flags, dtype=torch.float32, device=self.device
            ),
            "is_parseable": torch.tensor(
                parseable_flags, dtype=torch.float32, device=self.device
            ),
            "responses": responses,
            "prompts": prompts,
        }

    def _optimize(self, rollout: dict[str, Any]) -> dict[str, float]:
        assert self.model is not None
        assert self.value_head is not None
        assert self.optimizer is not None

        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]
        labels = rollout["labels"]
        old_logprobs = rollout["old_logprobs"]
        ref_logprobs = rollout["ref_logprobs"]
        old_values = rollout["old_values"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]
        action_mask = rollout["action_mask"]

        total = int(input_ids.size(0))
        minibatch_size = min(self.config.minibatch_size, total)

        total_losses: list[torch.Tensor] = []
        pg_losses: list[torch.Tensor] = []
        vf_losses: list[torch.Tensor] = []
        approx_kls: list[torch.Tensor] = []
        ref_kls: list[torch.Tensor] = []
        ratio_means: list[torch.Tensor] = []
        clip_fracs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        grad_norms: list[torch.Tensor] = []

        trainable_params = list(self.model.parameters()) + list(
            self.value_head.parameters()
        )

        self.model.train()
        self.value_head.train()

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                idx = permutation[start : start + minibatch_size]

                outputs = self.model(
                    input_ids=input_ids[idx],
                    attention_mask=attention_mask[idx],
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
                new_logprobs, _ = self._token_logprobs_from_logits(
                    outputs.logits,
                    labels[idx],
                )

                mb_action_mask = action_mask[idx]
                new_logprobs = new_logprobs * mb_action_mask
                mb_old_logprobs = old_logprobs[idx]
                mb_advantages = advantages[idx]

                log_ratio = (new_logprobs - mb_old_logprobs) * mb_action_mask
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )

                pg_losses1 = -mb_advantages * ratio
                pg_losses2 = -mb_advantages * clipped_ratio
                pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mb_action_mask)

                values = self.value_head(outputs.hidden_states[-1].float()).squeeze(-1)
                new_values = values[:, :-1] * mb_action_mask
                mb_old_values = old_values[idx]
                mb_returns = returns[idx]

                values_clipped = torch.clamp(
                    new_values,
                    mb_old_values - self.config.cliprange_value,
                    mb_old_values + self.config.cliprange_value,
                )
                vf_losses1 = (new_values - mb_returns).pow(2)
                vf_losses2 = (values_clipped - mb_returns).pow(2)
                vf_loss = 0.5 * masked_mean(
                    torch.max(vf_losses1, vf_losses2), mb_action_mask
                )

                token_entropy = self._token_entropy_from_logits(
                    outputs.logits[:, :-1, :]
                )
                entropy = masked_mean(token_entropy, mb_action_mask)

                loss = (
                    pg_loss
                    + (self.config.vf_coef * vf_loss)
                    - (self.config.ent_coef * entropy)
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = 0.5 * masked_mean(
                    (new_logprobs - mb_old_logprobs).pow(2), mb_action_mask
                )
                clip_frac = masked_mean(
                    (pg_losses2 > pg_losses1).float(), mb_action_mask
                )
                ratio_mean = masked_mean(ratio, mb_action_mask)

                total_losses.append(loss.detach())
                pg_losses.append(pg_loss.detach())
                vf_losses.append(vf_loss.detach())
                approx_kls.append(approx_kl.detach())
                clip_fracs.append(clip_frac.detach())
                ratio_means.append(ratio_mean.detach())
                entropies.append(entropy.detach())
                grad_norms.append(
                    torch.as_tensor(
                        float(grad_norm), dtype=torch.float32, device=self.device
                    )
                )

                if ref_logprobs is not None:
                    mb_ref_kl = masked_mean(
                        (new_logprobs - ref_logprobs[idx]), mb_action_mask
                    )
                    ref_kls.append(mb_ref_kl.detach())

        mean_ref_kl = torch.stack(ref_kls).mean().item() if ref_kls else 0.0
        self.kl_controller.update(
            current=mean_ref_kl, n_steps=self.config.prompts_per_step
        )

        reward_scores = rollout["scores"]
        token_rewards = rollout["rewards"]
        token_non_score = rollout["non_score_rewards"]
        adv_std = torch.sqrt(masked_var(advantages, action_mask, unbiased=False)).item()

        metrics = {
            "train/loss": torch.stack(total_losses).mean().item(),
            "train/policy_loss": torch.stack(pg_losses).mean().item(),
            "train/value_loss": torch.stack(vf_losses).mean().item(),
            "train/ratio_mean": torch.stack(ratio_means).mean().item(),
            "train/clip_frac": torch.stack(clip_fracs).mean().item(),
            "train/approx_kl": torch.stack(approx_kls).mean().item(),
            "train/ref_kl": mean_ref_kl,
            "train/kl_coef": self.kl_coef,
            "train/entropy": torch.stack(entropies).mean().item(),
            "train/grad_norm": torch.stack(grad_norms).mean().item(),
            "train/reward_mean": reward_scores.mean().item(),
            "train/reward_max": reward_scores.max().item(),
            "train/reward_min": reward_scores.min().item(),
            "train/token_reward_mean": masked_mean(token_rewards, action_mask).item(),
            "train/non_score_reward_mean": masked_mean(
                token_non_score, action_mask
            ).item(),
            "train/adv_std": adv_std,
            "train/correct_frac": rollout["is_correct"].mean().item(),
            "train/parseable_frac": rollout["is_parseable"].mean().item(),
            "train/has_answer_tag_frac": rollout["has_tag"].mean().item(),
            "train/response_len_mean": (action_mask.sum(dim=1).float().mean().item()),
        }

        if rollout["kls"] is not None:
            metrics["train/old_policy_ref_kl"] = masked_mean(
                rollout["kls"],
                action_mask,
            ).item()

        return metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, float]:
        assert self.model is not None

        self.model.eval()

        examples = self._build_eval_examples(step)
        prompts = [self._build_prompt(ex) for ex in examples]

        _, _, responses = self._generate_responses(
            prompts,
            do_sample=False,
            max_new_tokens=self.config.eval_max_new_tokens,
        )

        correct = 0
        parseable = 0
        has_tag = 0

        for example, response in zip(examples, responses):
            pred_answer = extract_answer(response)
            parseable += int(pred_answer is not None)
            has_tag += int("####" in response)
            correct += int(
                pred_answer is not None and pred_answer == example.final_answer
            )

        n = max(len(examples), 1)
        self.model.train()

        return {
            "eval/accuracy": correct / n,
            "eval/parseable_frac": parseable / n,
            "eval/has_answer_tag_frac": has_tag / n,
            "eval/return_max": correct / n,
        }

    def _save_checkpoint(self, out_dir: Path, step: int) -> None:
        assert self.model is not None
        assert self.value_head is not None
        assert self.tokenizer is not None

        checkpoint_dir = out_dir / f"lm_step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_dir = checkpoint_dir / "actor"
        tokenizer_dir = checkpoint_dir / "tokenizer"

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)

        torch.save(
            {
                "value_head_state_dict": self.value_head.state_dict(),
                "kl_coef": self.kl_coef,
                "config": asdict(self.config),
            },
            checkpoint_dir / "ppo_state.pt",
        )

    def train(self, out_dir: str) -> None:
        self._ensure_initialized()

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, self.config.num_steps + 1):
            examples = self._sample_train_examples()
            rollout = self._collect_rollout(examples)
            metrics = self._optimize(rollout)

            if step == 1 or step % self.config.eval_every == 0:
                metrics.update(self.evaluate(step=step))

            log_wandb(metrics, step=step, silent=True)

            if step == 1 or step % 10 == 0:
                sample_prompt = examples[0].question.replace("\n", " ").strip()
                sample_response = rollout["responses"][0].replace("\n", " ").strip()
                eval_acc = metrics.get("eval/accuracy")
                acc_display = (
                    f"{float(eval_acc):.3f}" if eval_acc is not None else "n/a"
                )
                print(
                    f"step={step} reward={metrics['train/reward_mean']:.3f} "
                    f"acc={acc_display} kl={metrics['train/kl_coef']:.4f} "
                    f"q='{sample_prompt[:80]}...' "
                    f"resp='{sample_response[:120]}...'"
                )

            if step % self.config.save_every == 0 or step == self.config.num_steps:
                self._save_checkpoint(output_dir, step=step)
