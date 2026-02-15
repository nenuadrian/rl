from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
except ImportError:  # pragma: no cover - exercised only when transformers is absent
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any


_INT_PATTERN = re.compile(r"([+-]?\d+)")


@dataclass(frozen=True)
class MathSample:
    prompt: str
    answer: str


def build_addition_dataset(
    num_samples: int,
    min_value: int = 0,
    max_value: int = 99,
    seed: int | None = None,
    prompt_template: str = (
        "Solve this addition problem and reply with only the final integer.\n"
        "Question: {a} + {b} = "
    ),
) -> list[MathSample]:
    """Create prompt/answer pairs for simple addition tasks."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")

    rng = random.Random(seed)
    samples: list[MathSample] = []
    for _ in range(num_samples):
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        prompt = prompt_template.format(a=a, b=b)
        samples.append(MathSample(prompt=prompt, answer=str(a + b)))
    return samples


def extract_first_integer(text: str) -> int | None:
    cleaned = str(text).strip()
    match = _INT_PATTERN.search(cleaned)
    if match is None:
        return None
    return int(match.group(1))


def extract_last_integer(text: str) -> int | None:
    # Backward-compatible alias for existing callers/tests.
    return extract_first_integer(text)


class ExactMathReward:
    """Exact-match integer reward for math responses."""

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        invalid_reward: float = 0.0,
    ):
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.invalid_reward = float(invalid_reward)

    def __call__(
        self,
        samples: Sequence[MathSample],
        responses: Sequence[str],
    ) -> torch.Tensor:
        if len(samples) != len(responses):
            raise ValueError("samples and responses must have the same length")

        scores: list[float] = []
        for sample, response in zip(samples, responses):
            target = extract_first_integer(sample.answer)
            prediction = extract_first_integer(response)
            if target is None or prediction is None:
                scores.append(self.invalid_reward)
            elif prediction == target:
                scores.append(self.correct_reward)
            else:
                scores.append(self.incorrect_reward)

        return torch.tensor(scores, dtype=torch.float32)


class ShapedMathReward:
    """Distance-shaped reward: 1 - |pred-target| / scale, clipped to [0, 1]."""

    def __init__(
        self,
        error_scale: float = 100.0,
        invalid_reward: float = 0.0,
    ):
        if error_scale <= 0:
            raise ValueError("error_scale must be > 0")
        self.error_scale = float(error_scale)
        self.invalid_reward = float(invalid_reward)

    def __call__(
        self,
        samples: Sequence[MathSample],
        responses: Sequence[str],
    ) -> torch.Tensor:
        if len(samples) != len(responses):
            raise ValueError("samples and responses must have the same length")

        scores: list[float] = []
        for sample, response in zip(samples, responses):
            target = extract_first_integer(sample.answer)
            prediction = extract_first_integer(response)
            if target is None or prediction is None:
                scores.append(self.invalid_reward)
                continue

            distance = abs(float(prediction - target))
            shaped = 1.0 - (distance / self.error_scale)
            scores.append(float(max(0.0, min(1.0, shaped))))

        return torch.tensor(scores, dtype=torch.float32)


@dataclass
class GPTPPOConfig:
    """Configuration for GPT-style PPO fine-tuning."""

    model_name: str = "distilgpt2"
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    learning_rate: float = 1e-5

    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0

    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.05
    normalize_advantages: bool = True

    max_prompt_length: int = 96
    max_response_length: int = 24
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    target_kl: float | None = None
    max_grad_norm: float = 1.0

    reward_type: str = "exact"
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    invalid_reward: float = 0.0
    reward_error_scale: float = 100.0

    sft_epochs: int = 0
    sft_batch_size: int = 0
    sft_learning_rate: float = 5e-5

    log_num_examples: int = 10
    eval_compare_modes: bool = True

    device: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be > 0")
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be > 0")
        if self.max_response_length <= 0:
            raise ValueError("max_response_length must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.clip_range < 0:
            raise ValueError("clip_range must be >= 0")
        if self.clip_range_value < 0:
            raise ValueError("clip_range_value must be >= 0")
        if self.reward_type not in {"exact", "shaped"}:
            raise ValueError("reward_type must be one of: exact, shaped")
        if self.reward_error_scale <= 0:
            raise ValueError("reward_error_scale must be > 0")
        if self.sft_epochs < 0:
            raise ValueError("sft_epochs must be >= 0")
        if self.sft_batch_size < 0:
            raise ValueError("sft_batch_size must be >= 0")
        if self.sft_learning_rate <= 0:
            raise ValueError("sft_learning_rate must be > 0")
        if self.log_num_examples < 0:
            raise ValueError("log_num_examples must be >= 0")


@dataclass
class PPORolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    ref_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    scores: torch.Tensor
    prompts: list[str]
    responses: list[str]


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (tensor * mask).sum() / denom


def compute_masked_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_advantages: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-token GAE only over masked (response) tokens."""
    if values.shape != rewards.shape or values.shape != action_mask.shape:
        raise ValueError("values, rewards, and action_mask must have matching shapes")

    bsz, time = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(bsz, dtype=values.dtype, device=values.device)

    for t in reversed(range(time)):
        if t == time - 1:
            next_values = torch.zeros_like(last_adv)
            next_mask = torch.zeros_like(last_adv)
        else:
            next_values = values[:, t + 1]
            next_mask = action_mask[:, t + 1]

        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        last_adv = delta + gamma * lam * next_mask * last_adv
        last_adv = last_adv * action_mask[:, t]
        advantages[:, t] = last_adv

    returns = advantages + values

    if normalize_advantages:
        valid_advantages = advantages[action_mask > 0]
        if valid_advantages.numel() > 1:
            mean = valid_advantages.mean()
            std = valid_advantages.std(unbiased=False).clamp(min=1e-6)
            advantages = torch.where(
                action_mask > 0,
                (advantages - mean) / std,
                torch.zeros_like(advantages),
            )

    return advantages, returns


class CausalLMWithValueHead(nn.Module):
    """A causal LM plus a token-value head for PPO."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Unable to infer hidden size from model config")

        self.value_head = nn.Linear(int(hidden_size), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states).squeeze(-1)

        result: dict[str, torch.Tensor] = {"logits": logits, "values": values}
        if return_entropy:
            shift_logits = logits[:, :-1, :]
            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            probs = log_probs.exp()
            result["entropy"] = -(probs * log_probs).sum(dim=-1)
        return result


class GPTPPOTrainer:
    """
    PPO trainer for decoder-only models (e.g. GPT-2 / DistilGPT2).

    This trainer is intentionally lightweight:
    - single-process / single-device
    - a frozen reference model for KL penalty
    - token-level PPO with a value head

    It is meant to be a practical starting point for math/addition fine-tuning.
    """

    def __init__(
        self,
        config: GPTPPOConfig,
        train_samples: Sequence[MathSample],
        reward_fn: Callable[[Sequence[MathSample], Sequence[str]], torch.Tensor]
        | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        actor_model: PreTrainedModel | None = None,
        reference_model: PreTrainedModel | None = None,
    ):
        if len(train_samples) == 0:
            raise ValueError("train_samples must not be empty")

        self.config = config
        self.train_samples = list(train_samples)
        if reward_fn is None:
            if config.reward_type == "shaped":
                reward_fn = ShapedMathReward(
                    error_scale=config.reward_error_scale,
                    invalid_reward=config.invalid_reward,
                )
            else:
                reward_fn = ExactMathReward(
                    correct_reward=config.correct_reward,
                    incorrect_reward=config.incorrect_reward,
                    invalid_reward=config.invalid_reward,
                )
        self.reward_fn = reward_fn
        self.device = self._resolve_device(config.device)
        self.global_step = 0
        self._did_sft_warmstart = False

        self._set_seed(config.seed)

        if tokenizer is None:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required to construct GPTPPOTrainer. "
                    "Install it with `pip install transformers`."
                )
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            else:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        if actor_model is None:
            if AutoModelForCausalLM is None:
                raise ImportError(
                    "transformers is required to construct GPTPPOTrainer. "
                    "Install it with `pip install transformers`."
                )
            actor_model = AutoModelForCausalLM.from_pretrained(config.model_name)

        if actor_model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            actor_model.resize_token_embeddings(len(self.tokenizer))

        self.policy = CausalLMWithValueHead(actor_model).to(self.device)

        if reference_model is None:
            reference_model = copy.deepcopy(actor_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        self.reference_model = reference_model.to(self.device)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=config.learning_rate)

    @staticmethod
    def _resolve_device(device_arg: str | None) -> torch.device:
        if device_arg is not None:
            return torch.device(device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _sample_batch(self) -> list[MathSample]:
        if len(self.train_samples) >= self.config.batch_size:
            return random.sample(self.train_samples, self.config.batch_size)
        return random.choices(self.train_samples, k=self.config.batch_size)

    def _generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_sample: bool,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        return self.policy.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_new_tokens=max_new_tokens or self.config.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @staticmethod
    def _compute_response_metrics(
        samples: Sequence[MathSample],
        responses: Sequence[str],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        parseable = 0
        exact_match = 0
        abs_errors: list[float] = []
        details: list[dict[str, Any]] = []

        for sample, response in zip(samples, responses):
            target = extract_first_integer(sample.answer)
            parsed = extract_first_integer(response)
            is_parseable = int(parsed is not None)
            is_exact = int(target is not None and parsed is not None and parsed == target)

            if parsed is not None and target is not None:
                abs_errors.append(float(abs(parsed - target)))

            parseable += is_parseable
            exact_match += is_exact
            details.append(
                {
                    "prompt": sample.prompt,
                    "target": target,
                    "response_raw": response,
                    "response_parsed": parsed,
                    "is_parseable": bool(is_parseable),
                    "is_exact_match": bool(is_exact),
                }
            )

        denom = max(1, len(samples))
        mean_abs_error = float(np.mean(abs_errors)) if abs_errors else float("nan")
        metrics = {
            "frac_parseable": float(parseable / denom),
            "frac_exact_match": float(exact_match / denom),
            "mean_abs_error": mean_abs_error,
        }
        return metrics, details

    def _log_examples(
        self,
        iteration: int,
        details: Sequence[dict[str, Any]],
        rewards: torch.Tensor,
        prefix: str,
    ) -> None:
        max_examples = min(int(self.config.log_num_examples), len(details))
        if max_examples <= 0:
            return

        indices = random.sample(range(len(details)), k=max_examples)
        reward_values = rewards.detach().cpu().tolist()
        for idx in indices:
            row = details[idx]
            print(
                f"[GPTPPO][{prefix}][example] "
                f"iter={iteration} idx={idx} "
                f"reward={float(reward_values[idx]):.4f} "
                f"parseable={row['is_parseable']} exact={row['is_exact_match']} "
                f"target={row['target']} parsed={row['response_parsed']} "
                f"prompt={row['prompt']!r} "
                f"response={row['response_raw']!r}"
            )

    def _sync_reference_to_policy(self) -> None:
        new_reference = copy.deepcopy(self.policy.model)
        new_reference.eval()
        for param in new_reference.parameters():
            param.requires_grad = False
        self.reference_model = new_reference.to(self.device)

    def run_sft_warmstart(self) -> list[float]:
        if self.config.sft_epochs <= 0:
            self._did_sft_warmstart = True
            return []
        if self._did_sft_warmstart:
            return []

        self.policy.train()

        sft_batch_size = (
            int(self.config.sft_batch_size)
            if self.config.sft_batch_size > 0
            else int(self.config.batch_size)
        )
        optimizer = optim.AdamW(
            self.policy.model.parameters(),
            lr=float(self.config.sft_learning_rate),
        )

        losses: list[float] = []
        for epoch in range(1, int(self.config.sft_epochs) + 1):
            indices = np.arange(len(self.train_samples))
            np.random.shuffle(indices)
            epoch_losses: list[float] = []

            for start in range(0, len(indices), sft_batch_size):
                end = min(start + sft_batch_size, len(indices))
                mb_indices = indices[start:end]
                mb_samples = [self.train_samples[int(i)] for i in mb_indices]

                prompts = [sample.prompt for sample in mb_samples]
                full_texts = [sample.prompt + sample.answer for sample in mb_samples]

                enc_full = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_prompt_length
                    + self.config.max_response_length,
                )
                enc_prompt = self.tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                )

                input_ids = enc_full["input_ids"].to(self.device)
                attention_mask = enc_full["attention_mask"].to(self.device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                seq_len = int(input_ids.size(1))
                for i, prompt_ids in enumerate(enc_prompt["input_ids"]):
                    non_pad_tokens = int(attention_mask[i].sum().item())
                    left_pad = seq_len - non_pad_tokens
                    prompt_len = min(int(len(prompt_ids)), non_pad_tokens)
                    labels[i, left_pad : left_pad + prompt_len] = -100

                outputs = self.policy.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                    use_cache=False,
                )
                loss = outputs.loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()

                step_loss = float(loss.detach().cpu().item())
                losses.append(step_loss)
                epoch_losses.append(step_loss)

            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(
                f"[GPTPPO][SFT] epoch={epoch}/{self.config.sft_epochs} "
                f"loss={mean_epoch_loss:.6f}"
            )

        self._sync_reference_to_policy()
        self._did_sft_warmstart = True
        return losses

    def _forward_policy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_entropy=return_entropy,
        )
        logits = outputs["logits"]
        values = outputs["values"]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        selected_logits = torch.gather(
            shift_logits,
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        log_norm = torch.logsumexp(shift_logits, dim=-1)
        token_logprobs = selected_logits.float() - log_norm.float()

        token_values = values[:, :-1].float()

        entropy = outputs.get("entropy")
        if entropy is not None:
            entropy = entropy.float()

        return token_logprobs, token_values, entropy

    def _forward_reference_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        selected_logits = torch.gather(
            shift_logits,
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        log_norm = torch.logsumexp(shift_logits, dim=-1)
        return selected_logits.float() - log_norm.float()

    def collect_rollout(self, samples: Sequence[MathSample]) -> PPORolloutBatch:
        prompts = [sample.prompt for sample in samples]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_prompt_length,
        )

        prompt_input_ids = encoded["input_ids"].to(self.device)
        prompt_attention_mask = encoded["attention_mask"].to(self.device)
        prompt_width = int(prompt_input_ids.size(1))
        if prompt_width <= 0:
            raise RuntimeError("Tokenizer produced empty prompts")

        self.policy.eval()
        with torch.no_grad():
            generated_ids = self._generate_batch(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                do_sample=True,
                max_new_tokens=self.config.max_response_length,
            )

        batch_size = int(generated_ids.size(0))
        seq_len = int(generated_ids.size(1))

        full_attention_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=prompt_attention_mask.dtype,
            device=self.device,
        )
        full_attention_mask[:, :prompt_width] = prompt_attention_mask

        action_mask = torch.zeros(
            (batch_size, seq_len - 1),
            dtype=torch.float32,
            device=self.device,
        )

        responses: list[str] = []
        response_lengths: list[int] = []

        max_possible_response = max(1, seq_len - prompt_width)
        max_response = min(self.config.max_response_length, max_possible_response)

        eos_id = self.tokenizer.eos_token_id
        for i in range(batch_size):
            response_tokens = generated_ids[i, prompt_width : prompt_width + max_response]
            response_len = int(response_tokens.numel())

            if eos_id is not None and response_len > 0:
                eos_positions = (response_tokens == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    response_len = int(eos_positions[0].item()) + 1

            response_len = max(1, min(response_len, max_response))
            response_lengths.append(response_len)

            full_attention_mask[i, prompt_width : prompt_width + response_len] = 1

            action_start = prompt_width - 1
            action_end = action_start + response_len
            action_mask[i, action_start:action_end] = 1.0

            response_text = self.tokenizer.decode(
                response_tokens[:response_len],
                skip_special_tokens=True,
            ).strip()
            responses.append(response_text)

        with torch.no_grad():
            old_logprobs, old_values, _ = self._forward_policy(
                generated_ids,
                full_attention_mask,
                return_entropy=False,
            )
            ref_logprobs = self._forward_reference_logprobs(
                generated_ids,
                full_attention_mask,
            )

        scores = self.reward_fn(samples, responses)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)
        scores = scores.to(device=self.device, dtype=torch.float32).view(-1)
        if scores.numel() != batch_size:
            raise ValueError(
                f"Reward fn returned {scores.numel()} scores for {batch_size} samples"
            )

        kl = old_logprobs - ref_logprobs
        non_score_rewards = -self.config.kl_coef * kl

        rewards = non_score_rewards * action_mask
        for i, response_len in enumerate(response_lengths):
            last_action_idx = (prompt_width - 1) + response_len - 1
            rewards[i, last_action_idx] += scores[i]

        advantages, returns = compute_masked_gae(
            values=old_values,
            rewards=rewards,
            action_mask=action_mask,
            gamma=self.config.gamma,
            lam=self.config.lam,
            normalize_advantages=self.config.normalize_advantages,
        )

        return PPORolloutBatch(
            input_ids=generated_ids,
            attention_mask=full_attention_mask,
            action_mask=action_mask,
            old_logprobs=old_logprobs.detach(),
            old_values=old_values.detach(),
            ref_logprobs=ref_logprobs.detach(),
            rewards=rewards.detach(),
            advantages=advantages.detach(),
            returns=returns.detach(),
            scores=scores.detach(),
            prompts=prompts,
            responses=responses,
        )

    def ppo_update(self, rollout: PPORolloutBatch) -> dict[str, float]:
        self.policy.train()

        batch_size = int(rollout.input_ids.size(0))
        indices = np.arange(batch_size)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []

        early_stop = False

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_idx = torch.as_tensor(indices[start:end], device=self.device)

                input_ids = rollout.input_ids[mb_idx]
                attention_mask = rollout.attention_mask[mb_idx]
                action_mask = rollout.action_mask[mb_idx]

                old_logprobs = rollout.old_logprobs[mb_idx]
                old_values = rollout.old_values[mb_idx]
                advantages = rollout.advantages[mb_idx]
                returns = rollout.returns[mb_idx]

                new_logprobs, new_values, entropy = self._forward_policy(
                    input_ids,
                    attention_mask,
                    return_entropy=True,
                )
                if entropy is None:
                    raise RuntimeError("Expected entropy tensor when return_entropy=True")

                log_ratio = (new_logprobs - old_logprobs) * action_mask
                ratio = torch.exp(log_ratio)

                pg_losses1 = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                policy_loss = masked_mean(torch.max(pg_losses1, pg_losses2), action_mask)

                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.clip_range_value,
                    self.config.clip_range_value,
                )
                value_losses1 = (new_values - returns) ** 2
                value_losses2 = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * masked_mean(
                    torch.max(value_losses1, value_losses2),
                    action_mask,
                )

                entropy_loss = masked_mean(entropy, action_mask)

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = masked_mean(old_logprobs - new_logprobs, action_mask)
                    clipfrac = masked_mean(
                        (torch.abs(ratio - 1.0) > self.config.clip_range).float(),
                        action_mask,
                    )

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_values.append(float(entropy_loss.detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))
                clipfracs.append(float(clipfrac.detach().cpu().item()))

                if (
                    self.config.target_kl is not None
                    and float(approx_kl.detach().cpu().item()) > float(self.config.target_kl)
                ):
                    early_stop = True
                    break
            if early_stop:
                break

        self.global_step += 1

        score_mean = float(rollout.scores.mean().detach().cpu().item())
        score_std = float(rollout.scores.std(unbiased=False).detach().cpu().item())
        valid_advantages = rollout.advantages[rollout.action_mask > 0]
        adv_variance = (
            float(valid_advantages.var(unbiased=False).detach().cpu().item())
            if valid_advantages.numel() > 0
            else 0.0
        )

        metrics = {
            "train/step": float(self.global_step),
            "train/policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "train/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "train/entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "train/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "train/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "train/clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "train/score_mean": score_mean,
            "train/score_std": score_std,
            "train/adv_variance": adv_variance,
        }
        return metrics

    def train(
        self,
        num_iterations: int,
        save_dir: str | Path | None = None,
        save_every: int = 0,
        log_every: int = 1,
    ) -> list[dict[str, float]]:
        if num_iterations <= 0:
            raise ValueError("num_iterations must be > 0")

        if self.config.sft_epochs > 0 and not self._did_sft_warmstart:
            self.run_sft_warmstart()

        history: list[dict[str, float]] = []
        for iteration in range(1, num_iterations + 1):
            batch = self._sample_batch()
            rollout = self.collect_rollout(batch)
            metrics = self.ppo_update(rollout)

            sampled_metrics, sampled_details = self._compute_response_metrics(
                batch, rollout.responses
            )
            metrics["train/frac_parseable"] = sampled_metrics["frac_parseable"]
            metrics["train/frac_exact_match"] = sampled_metrics["frac_exact_match"]
            metrics["train/mean_abs_error"] = sampled_metrics["mean_abs_error"]
            metrics["eval_sampled/frac_parseable"] = sampled_metrics["frac_parseable"]
            metrics["eval_sampled/frac_exact_match"] = sampled_metrics["frac_exact_match"]
            metrics["eval_sampled/mean_abs_error"] = sampled_metrics["mean_abs_error"]

            self._log_examples(
                iteration=iteration,
                details=sampled_details,
                rewards=rollout.scores,
                prefix="sampled",
            )

            if self.config.eval_compare_modes:
                greedy_responses = self.generate(
                    [sample.prompt for sample in batch],
                    deterministic=True,
                    max_new_tokens=self.config.max_response_length,
                )
                greedy_metrics, greedy_details = self._compute_response_metrics(
                    batch, greedy_responses
                )
                metrics["eval_greedy/frac_parseable"] = greedy_metrics["frac_parseable"]
                metrics["eval_greedy/frac_exact_match"] = greedy_metrics["frac_exact_match"]
                metrics["eval_greedy/mean_abs_error"] = greedy_metrics["mean_abs_error"]

                greedy_rewards = self.reward_fn(batch, greedy_responses)
                if not isinstance(greedy_rewards, torch.Tensor):
                    greedy_rewards = torch.tensor(greedy_rewards, dtype=torch.float32)
                greedy_rewards = greedy_rewards.to(torch.float32)
                self._log_examples(
                    iteration=iteration,
                    details=greedy_details,
                    rewards=greedy_rewards,
                    prefix="greedy",
                )

            history.append(metrics)

            if log_every > 0 and iteration % log_every == 0:
                print(
                    "[GPTPPO] "
                    f"iter={iteration} "
                    f"score_mean={metrics['train/score_mean']:.4f} "
                    f"frac_parseable={metrics['train/frac_parseable']:.3f} "
                    f"frac_exact={metrics['train/frac_exact_match']:.3f} "
                    f"mae={metrics['train/mean_abs_error']:.3f} "
                    f"policy_loss={metrics['train/policy_loss']:.4f} "
                    f"value_loss={metrics['train/value_loss']:.4f} "
                    f"approx_kl={metrics['train/approx_kl']:.6f} "
                    f"clip_fraction={metrics['train/clip_fraction']:.4f} "
                    f"adv_var={metrics['train/adv_variance']:.6f}"
                )
                if self.config.eval_compare_modes:
                    print(
                        "[GPTPPO][eval] "
                        f"iter={iteration} "
                        f"sampled_exact={metrics['eval_sampled/frac_exact_match']:.3f} "
                        f"greedy_exact={metrics['eval_greedy/frac_exact_match']:.3f} "
                        f"sampled_parseable={metrics['eval_sampled/frac_parseable']:.3f} "
                        f"greedy_parseable={metrics['eval_greedy/frac_parseable']:.3f}"
                    )

            if save_dir is not None and save_every > 0 and iteration % save_every == 0:
                ckpt_dir = Path(save_dir) / f"step_{iteration:06d}"
                self.save_pretrained(ckpt_dir)

        return history

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int | None = None,
        deterministic: bool = True,
    ) -> list[str]:
        if len(prompts) == 0:
            return []

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_prompt_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_width = int(input_ids.size(1))

        self.policy.eval()
        with torch.no_grad():
            generated_ids = self._generate_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=not deterministic,
                max_new_tokens=max_new_tokens or self.config.max_response_length,
            )

        responses: list[str] = []
        for row in generated_ids:
            response_tokens = row[prompt_width:]
            responses.append(
                self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            )
        return responses

    def save_pretrained(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        policy_dir = out / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)

        self.policy.model.save_pretrained(policy_dir)
        self.tokenizer.save_pretrained(policy_dir)

        torch.save(self.policy.value_head.state_dict(), out / "value_head.pt")

        with (out / "gpt_ppo_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
