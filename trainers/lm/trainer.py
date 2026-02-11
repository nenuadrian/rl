from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.wandb_utils import log_wandb


@dataclass(frozen=True)
class ArithmeticProblem:
    a: int
    b: int
    answer: int


@dataclass
class LMGRPOConfig:
    model_name: str
    dtype: str = "bfloat16"
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    num_steps: int = 300
    prompts_per_step: int = 8
    group_size: int = 4
    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    max_new_tokens: int = 24
    temperature: float = 0.9
    top_k: int = 40
    eval_every: int = 25
    eval_examples: int = 64
    save_every: int = 100
    train_min_operand: int = 0
    train_max_operand: int = 30
    eval_min_operand: int = 0
    eval_max_operand: int = 30
    reward_std_eps: float = 1e-6
    advantage_mode: str = "ema_baseline"
    normalize_advantages: bool = True
    baseline_momentum: float = 0.9
    kl_coef: float = 0.02
    target_ref_kl: float = 0.08
    kl_adaptation_factor: float = 1.5
    seed: int = 0


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
    """Single-GPU GRPO trainer for arithmetic fine-tuning of causal LMs."""

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
        self.tokenizer: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.kl_coef: float = self.config.kl_coef
        self.reward_baseline: float | None = None

    def _ensure_initialized(self) -> None:
        if (
            self.model is not None
            and self.reference_model is not None
            and self.tokenizer is not None
            and self.optimizer is not None
        ):
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        # Decoder-only generation is safest with left padding.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer must define either a pad token or EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token

        model_dtype = _resolve_dtype(self.config.dtype, self.device)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=model_dtype,
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=model_dtype,
        )
        model.to(self.device)
        reference_model.to(self.device)
        model.train()
        reference_model.eval()
        reference_model.requires_grad_(False)
        model.config.use_cache = False
        reference_model.config.use_cache = False

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    @staticmethod
    def _build_prompt(problem: ArithmeticProblem) -> str:
        return (
            "Solve the addition problem.\n"
            "Reply with exactly one line in this format:\n"
            f"So {problem.a} + {problem.b} = <answer>\n"
            "Answer:\n"
        )

    @staticmethod
    def _parse_answer(problem: ArithmeticProblem, text: str) -> int | None:
        pattern = rf"So\s*{problem.a}\s*\+\s*{problem.b}\s*=\s*(-?\d+)"
        match = re.search(pattern, text)
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _is_correct(cls, problem: ArithmeticProblem, text: str) -> bool:
        parsed = cls._parse_answer(problem, text)
        return parsed == problem.answer

    @classmethod
    def _reward(cls, problem: ArithmeticProblem, text: str) -> float:
        prefix = f"So {problem.a} + {problem.b} = "
        reward = 0.0
        if prefix in text:
            reward += 0.2

        parsed = cls._parse_answer(problem, text)
        if parsed is not None:
            delta = abs(parsed - problem.answer)
            reward += 0.8 / (1.0 + float(delta))
            if parsed == problem.answer:
                reward = 1.0
        return min(reward, 1.0)

    def _sample_problem(self, min_operand: int, max_operand: int) -> ArithmeticProblem:
        a = self.train_rng.randint(min_operand, max_operand)
        b = self.train_rng.randint(min_operand, max_operand)
        return ArithmeticProblem(a=a, b=b, answer=a + b)

    def _sample_train_problems(self) -> list[ArithmeticProblem]:
        assert self.config is not None
        return [
            self._sample_problem(
                min_operand=self.config.train_min_operand,
                max_operand=self.config.train_max_operand,
            )
            for _ in range(self.config.prompts_per_step)
        ]

    def _build_eval_problems(self, step: int) -> list[ArithmeticProblem]:
        assert self.config is not None
        rng = random.Random(self.config.seed + (step * 104729))
        problems = [ArithmeticProblem(a=10, b=3, answer=13)]
        for _ in range(max(0, self.config.eval_examples - 1)):
            a = rng.randint(self.config.eval_min_operand, self.config.eval_max_operand)
            b = rng.randint(self.config.eval_min_operand, self.config.eval_max_operand)
            problems.append(ArithmeticProblem(a=a, b=b, answer=a + b))
        return problems

    def _sequence_log_probs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        token_log_probs = token_log_probs.squeeze(-1)
        masked_log_probs = token_log_probs * completion_mask
        token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
        return masked_log_probs.sum(dim=1) / token_counts

    def _tokenize_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in encoded.items()}

    @staticmethod
    def _completion_start_index(encoded: dict[str, torch.Tensor]) -> int:
        # HF generate appends new tokens after the full padded input width.
        return int(encoded["input_ids"].size(1))

    @staticmethod
    def _split_completions(
        generated: torch.Tensor,
        completion_start: int,
    ) -> torch.Tensor:
        if generated.size(1) < completion_start:
            raise ValueError(
                f"Generated sequence length {generated.size(1)} is smaller than "
                f"completion start index {completion_start}."
            )
        return generated[:, completion_start:]

    def _collect_rollout(
        self, problems: list[ArithmeticProblem]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        assert self.config is not None
        assert self.model is not None
        assert self.reference_model is not None
        assert self.tokenizer is not None

        repeated_problems: list[ArithmeticProblem] = []
        prompts: list[str] = []
        for problem in problems:
            prompt = self._build_prompt(problem)
            for _ in range(self.config.group_size):
                repeated_problems.append(problem)
                prompts.append(prompt)

        encoded = self._tokenize_prompts(prompts)
        completion_start = self._completion_start_index(encoded)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=True,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = self._split_completions(generated, completion_start)
        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        rewards = torch.tensor(
            [
                self._reward(problem, text)
                for problem, text in zip(repeated_problems, completion_texts)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._compute_advantages(
            rewards=rewards,
            num_problems=len(problems),
        ).detach()

        generated_mask = torch.ones(
            generated.size(0),
            self.config.max_new_tokens,
            dtype=encoded["attention_mask"].dtype,
            device=self.device,
        )
        attention_mask = torch.cat([encoded["attention_mask"], generated_mask], dim=1)
        completion_mask = torch.zeros(
            generated.size(0),
            generated.size(1) - 1,
            dtype=torch.float32,
            device=self.device,
        )
        completion_mask[:, completion_start - 1 :] = 1.0

        with torch.no_grad():
            old_seq_log_probs = self._sequence_log_probs(
                self.model,
                input_ids=generated,
                attention_mask=attention_mask,
                completion_mask=completion_mask,
            ).detach()
            ref_seq_log_probs = self._sequence_log_probs(
                self.reference_model,
                input_ids=generated,
                attention_mask=attention_mask,
                completion_mask=completion_mask,
            ).detach()

        rollout = {
            "input_ids": generated.detach(),
            "attention_mask": attention_mask.detach(),
            "completion_mask": completion_mask.detach(),
            "old_seq_log_probs": old_seq_log_probs,
            "ref_seq_log_probs": ref_seq_log_probs,
            "advantages": advantages,
            "rewards": rewards.detach(),
        }
        return rollout, completion_texts

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        num_problems: int,
    ) -> torch.Tensor:
        assert self.config is not None
        mode = self.config.advantage_mode

        if mode == "group_norm":
            rewards_view = rewards.view(num_problems, self.config.group_size)
            group_mean = rewards_view.mean(dim=1, keepdim=True)
            group_std = rewards_view.std(dim=1, keepdim=True, unbiased=False)
            return (
                (rewards_view - group_mean) / (group_std + self.config.reward_std_eps)
            ).reshape(-1)

        if mode == "group_centered":
            rewards_view = rewards.view(num_problems, self.config.group_size)
            group_mean = rewards_view.mean(dim=1, keepdim=True)
            advantages = (rewards_view - group_mean).reshape(-1)
            if self.config.normalize_advantages:
                std = advantages.std(unbiased=False)
                advantages = advantages / (std + self.config.reward_std_eps)
            return advantages

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
            "Use one of: raw, batch_norm, ema_baseline, group_centered, group_norm."
        )

    def _optimize(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        assert self.config is not None
        assert self.model is not None
        assert self.optimizer is not None

        input_ids = rollout["input_ids"]
        attention_mask = rollout["attention_mask"]
        completion_mask = rollout["completion_mask"]
        old_seq_log_probs = rollout["old_seq_log_probs"]
        ref_seq_log_probs = rollout["ref_seq_log_probs"]
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

        self.model.train()
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                idx = permutation[start : start + minibatch_size]

                seq_log_probs = self._sequence_log_probs(
                    self.model,
                    input_ids=input_ids[idx],
                    attention_mask=attention_mask[idx],
                    completion_mask=completion_mask[idx],
                )
                old_log_ratio = seq_log_probs - old_seq_log_probs[idx]
                ratios = torch.exp(old_log_ratio)
                clipped_ratios = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_objective = torch.min(
                    ratios * advantages[idx], clipped_ratios * advantages[idx]
                )
                ref_log_ratio = seq_log_probs - ref_seq_log_probs[idx]
                # Non-negative approximation to KL(current || reference).
                ref_kl = torch.exp(ref_log_ratio) - 1.0 - ref_log_ratio
                objective = policy_objective - (self.kl_coef * ref_kl)
                loss = -objective.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                approx_kl = 0.5 * old_log_ratio.pow(2).mean()
                losses.append(loss.detach())
                ratio_means.append(ratios.mean().detach())
                approx_kls.append(approx_kl.detach())
                ref_kls.append(ref_kl.mean().detach())
                grad_norms.append(
                    torch.as_tensor(grad_norm, dtype=torch.float32, device=self.device)
                )

        mean_loss = torch.stack(losses).mean().item()
        mean_ratio = torch.stack(ratio_means).mean().item()
        mean_kl = torch.stack(approx_kls).mean().item()
        mean_ref_kl = torch.stack(ref_kls).mean().item()
        mean_grad_norm = torch.stack(grad_norms).mean().item()
        reward_mean = rewards.mean().item()
        reward_max = rewards.max().item()
        reward_min = rewards.min().item()
        advantage_std = advantages.std(unbiased=False).item()

        if mean_ref_kl > (self.config.target_ref_kl * self.config.kl_adaptation_factor):
            self.kl_coef *= 1.1
        elif mean_ref_kl < (
            self.config.target_ref_kl / self.config.kl_adaptation_factor
        ):
            self.kl_coef /= 1.1
        self.kl_coef = float(max(self.kl_coef, 1e-6))

        return {
            "train/loss": mean_loss,
            "train/ratio_mean": mean_ratio,
            "train/approx_kl": mean_kl,
            "train/ref_kl": mean_ref_kl,
            "train/kl_coef": self.kl_coef,
            "train/grad_norm": mean_grad_norm,
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
        assert self.config is not None
        assert self.model is not None
        assert self.tokenizer is not None

        self.model.eval()
        problems = self._build_eval_problems(step)
        prompts = [self._build_prompt(problem) for problem in problems]
        encoded = self._tokenize_prompts(prompts)
        completion_start = self._completion_start_index(encoded)

        generated = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion_ids = self._split_completions(generated, completion_start)
        completions = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        total = len(problems)
        num_correct = sum(
            1
            for problem, text in zip(problems, completions)
            if self._is_correct(problem, text)
        )
        num_format = sum(
            1
            for problem, text in zip(problems, completions)
            if f"So {problem.a} + {problem.b} = " in text
        )
        ten_plus_three_correct = (
            1.0 if self._is_correct(problems[0], completions[0]) else 0.0
        )

        self.model.train()
        return {
            "eval/accuracy": float(num_correct) / float(total),
            "eval/format_rate": float(num_format) / float(total),
            "eval/ten_plus_three_correct": ten_plus_three_correct,
        }

    def _save_checkpoint(self, out_dir: Path, step: int) -> None:
        assert self.model is not None
        assert self.tokenizer is not None
        checkpoint_dir = out_dir / f"lm_step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def train(self, out_dir: str) -> None:
        self._ensure_initialized()
        assert self.config is not None

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, self.config.num_steps + 1):
            problems = self._sample_train_problems()
            rollout, sampled_texts = self._collect_rollout(problems)
            metrics = self._optimize(rollout)

            if step == 1 or step % self.config.eval_every == 0:
                metrics.update(self.evaluate(step=step))

            log_wandb(metrics, step=step, silent=True)

            if step == 1 or step % 10 == 0:
                sample_problem = problems[0]
                sample_text = sampled_texts[0].replace("\n", " ").strip()
                print(
                    f"step={step} reward={metrics['train/reward_mean']:.3f} "
                    f"acc={metrics.get('eval/accuracy', float('nan')):.3f} "
                    f"sample='So {sample_problem.a} + {sample_problem.b} = ...' "
                    f"model_output='{sample_text[:120]}'"
                )

            if step % self.config.save_every == 0 or step == self.config.num_steps:
                self._save_checkpoint(out_dir=output_dir, step=step)
