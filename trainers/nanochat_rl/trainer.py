import torch
import itertools
import json
import wandb
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from nanochat.tasks.gsm8k import GSM8K, extract_answer
from .agent import ChatRLAgent, ChatRLConfig

from nanochat.checkpoint_manager import save_checkpoint


class ChatRLTrainer:
    def __init__(self, agent: ChatRLAgent, config: ChatRLConfig, device):
        self.agent = agent
        self.config = config
        self.device = device
        self.train_task = GSM8K(subset="main", split="train")
        self.val_task = GSM8K(subset="main", split="test")
        self.tokenizer = agent.tokenizer
        self.engine = agent.engine
        self.num_steps = (
            len(self.train_task) // config.examples_per_step
        ) * config.num_epochs
        device_type = getattr(device, "type", None)
        device_type = device_type if device_type is not None else str(device)
        self.ptdtype = (
            torch.float32
            if getattr(config, "dtype", "bfloat16") == "float32"
            else torch.bfloat16
        )
        self.autocast_ctx = (
            torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
            if device_type == "cuda"
            else nullcontext()
        )
        # current training step (exposed to get_batch for deterministic seeding)
        self._current_step = 0

    def _compute_optim_stats(self):
        """Collect optimizer-adjacent stats to track if gradients/weights are changing."""
        grad_sq_sum = 0.0
        grad_abs_max = 0.0
        grad_nonzero_elems = 0
        grad_total_elems = 0
        grad_param_count = 0
        param_sq_sum = 0.0
        param_abs_max = 0.0
        param_total_elems = 0

        for param in self.agent.model.parameters():
            param_data = param.detach().float()
            param_sq_sum += param_data.square().sum().item()
            param_abs_max = max(param_abs_max, param_data.abs().max().item())
            param_total_elems += param.numel()

            if param.grad is None:
                continue

            grad_param_count += 1
            grad = param.grad.detach().float()
            grad_sq_sum += grad.square().sum().item()
            grad_abs_max = max(grad_abs_max, grad.abs().max().item())
            grad_nonzero_elems += torch.count_nonzero(grad).item()
            grad_total_elems += grad.numel()

        grad_global_norm = grad_sq_sum**0.5
        param_global_norm = param_sq_sum**0.5
        grad_rms = (grad_sq_sum / max(grad_total_elems, 1)) ** 0.5
        param_rms = (param_sq_sum / max(param_total_elems, 1)) ** 0.5
        grad_nonzero_frac = grad_nonzero_elems / max(grad_total_elems, 1)

        return {
            "optim/grad_global_norm": grad_global_norm,
            "optim/grad_rms": grad_rms,
            "optim/grad_abs_max": grad_abs_max,
            "optim/grad_nonzero_frac": grad_nonzero_frac,
            "optim/grad_param_count": float(grad_param_count),
            "optim/param_global_norm": param_global_norm,
            "optim/param_rms": param_rms,
            "optim/param_abs_max": param_abs_max,
            "optim/grad_to_param_norm_ratio": grad_global_norm
            / max(param_global_norm, 1e-12),
        }

    def get_batch(self):
        train_task = self.train_task
        tokenizer = self.tokenizer
        device = self.device
        engine = self.engine
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        example_indices = range(0, len(train_task))
        for example_idx in itertools.cycle(example_indices):
            agent = self.agent
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            generated_token_sequences = []
            generated_texts = []
            masks = []
            # Match scripts/chat_rl.py: always sample in eval mode.
            agent.eval_mode()
            num_sampling_steps = (
                self.config.num_samples // self.config.device_batch_size
            )
            for sampling_step in range(num_sampling_steps):
                print(
                    f"Sampling step {sampling_step + 1}/{num_sampling_steps} for example {example_idx}"
                )
                seed = (
                    hash((self._current_step, example_idx, sampling_step)) & 0x7FFFFFFF
                )
                with self.autocast_ctx:
                    batch, batch_masks = engine.generate_batch(
                        tokens,
                        num_samples=self.config.device_batch_size,
                        max_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        seed=seed,
                    )
                print(f"Generated batch of {len(batch)} samples")
                generated_token_sequences.extend(batch)
                masks.extend(batch_masks)
            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                generated_texts.append(generated_text)
                reward = train_task.reward(conversation, generated_text)
                rewards.append(reward)
            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_generated_token_sequences = [
                seq + [assistant_end] * (max_length - len(seq))
                for seq in generated_token_sequences
            ]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
            ids = torch.tensor(
                padded_generated_token_sequences, dtype=torch.long, device=device
            )
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            mu = rewards.mean()
            advantages = rewards - mu
            yield generated_token_sequences, generated_texts, inputs, targets, rewards, advantages

    def run_gsm8k_eval(
        self,
        max_examples=None,
        num_samples=1,
        max_completion_tokens=256,
        temperature=0.0,
        top_k=50,
    ):
        val_task = self.val_task
        tokenizer = self.tokenizer
        engine = self.engine
        max_examples = (
            min(max_examples, len(val_task))
            if max_examples is not None
            else len(val_task)
        )
        print(
            f"Running GSM8K evaluation on {max_examples} examples with {num_samples} samples each..."
        )
        for idx in range(0, max_examples):
            conversation = val_task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            with self.autocast_ctx:
                generated_token_sequences, masks = engine.generate_batch(
                    tokens,
                    num_samples=num_samples,
                    max_tokens=max_completion_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            outcomes = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                is_correct = val_task.evaluate(conversation, generated_text)
                outcomes.append({"is_correct": is_correct})
            record = {"idx": idx, "outcomes": outcomes}
            print(record)
            yield record

    @torch.no_grad()
    def run_evaluation(
        self,
        out_dir: str | None = None,
        *,
        step: int = 0,
    ) -> dict[str, float]:
        val_task = self.val_task
        tokenizer = self.tokenizer
        engine = self.engine
        max_examples = min(self.config.eval_examples, len(val_task))
        if max_examples <= 0:
            raise ValueError("eval_examples must be > 0 for evaluation-only mode")

        num_samples = max(1, int(self.config.num_samples))
        device_batch_size = max(1, int(self.config.device_batch_size))
        num_sampling_steps = (num_samples + device_batch_size - 1) // device_batch_size
        progress_log_every = max(1, max_examples // 100)
        text_rows: list[list[object]] = []
        max_text_rows = min(max_examples, 512)
        self.agent.eval_mode()
        print(
            f"Running GSM8K evaluation on {max_examples} examples with {num_samples} samples each..."
        )

        num_passed = 0
        total = 0
        for idx in range(max_examples):
            conversation = val_task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            generated_token_sequences = []
            for _ in range(num_sampling_steps):
                samples_remaining = num_samples - len(generated_token_sequences)
                step_samples = min(device_batch_size, samples_remaining)
                with self.autocast_ctx:
                    generated_batch, _ = engine.generate_batch(
                        tokens,
                        num_samples=step_samples,
                        max_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                    )
                generated_token_sequences.extend(generated_batch)
            completions = [
                tokenizer.decode(sample_tokens[prefix_length:])
                for sample_tokens in generated_token_sequences
            ]
            outcomes = [
                val_task.evaluate(conversation, completion)
                for completion in completions
            ]
            passed = any(outcomes)
            question_text = str(conversation["messages"][0]["content"])
            first_response = completions[0] if completions else ""
            first_correct_response = next(
                (completion for completion, outcome in zip(completions, outcomes) if outcome),
                "",
            )

            if len(text_rows) < max_text_rows:
                text_rows.append(
                    [
                        int(idx),
                        question_text,
                        first_response,
                        first_correct_response,
                        int(passed),
                    ]
                )

            total += 1
            num_passed += int(passed)
            print(
                f"\r\033[K{num_passed}/{total} ({100 * num_passed / total:.2f}%)",
                end="",
                flush=True,
            )
            if total % progress_log_every == 0 or total == max_examples:
                running_accuracy = num_passed / total
                wandb.log(
                    {
                        "step": step,
                        "eval/progress_accuracy": float(running_accuracy),
                        "eval/progress_num_passed": float(num_passed),
                        "eval/progress_num_total": float(total),
                        "eval/progress_frac_complete": float(total / max_examples),
                        "eval/latest_idx": int(idx),
                        "eval/latest_question": question_text[:1024],
                        "eval/latest_response": first_response[:2048],
                        "eval/latest_correct_response": first_correct_response[:2048],
                        "eval/latest_passed": int(passed),
                    }
                )

        print()
        accuracy = num_passed / total
        print(f"Final: {num_passed}/{total} ({100 * accuracy:.2f}%)")

        metrics = {
            "eval/accuracy": float(accuracy),
            "eval/return_max": float(accuracy),
            "eval/num_passed": float(num_passed),
            "eval/num_total": float(total),
            "eval/num_samples": float(num_samples),
        }
        wandb.log({"step": step, **metrics})
        if text_rows:
            wandb.log(
                {
                    "step": step,
                    "eval/text_samples": wandb.Table(
                        columns=[
                            "idx",
                            "question",
                            "first_response",
                            "first_correct_response",
                            "passed",
                        ],
                        data=text_rows,
                    ),
                }
            )

        if out_dir is not None:
            output_dir = Path(out_dir)
            eval_dir = output_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = eval_dir / f"metrics_step_{step:06d}.json"
            payload = {
                "step": int(step),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "config": asdict(self.config),
            }
            metrics_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Saved evaluation metrics to {metrics_path}")

        return metrics

    def train(self, out_dir: str):
        if self.config.run_evaluation:
            self.run_evaluation(out_dir=out_dir, step=0)
            return

        agent = self.agent
        device = self.device
        num_steps = self.num_steps
        examples_per_step = self.config.examples_per_step
        batch_iterator = self.get_batch()
        for step in range(num_steps):
            # expose current step for deterministic sampling seeds inside get_batch
            self._current_step = step
            print(f"Starting step {step}/{num_steps}...")
            if step % self.config.eval_every == 0:
                print("Running evaluation...")
                agent.eval_mode()
                passk = torch.zeros(self.config.device_batch_size, device=device)
                records_iter = self.run_gsm8k_eval(
                    max_examples=self.config.eval_examples,
                    num_samples=self.config.device_batch_size,
                    temperature=1.0,
                )
                records = list(records_iter)
                for k in range(1, self.config.device_batch_size + 1):
                    passk[k - 1] = sum(
                        any(o["is_correct"] for o in r["outcomes"][:k]) for r in records
                    )
                num_records = torch.tensor(
                    len(records), dtype=torch.long, device=device
                )
                passk = passk / num_records.item()
                print_passk = [
                    f"Pass@{k}: {passk[k - 1].item():.4f}"
                    for k in range(1, self.config.device_batch_size + 1)
                ]
                print(f"Step {step} | {', '.join(print_passk)}")
                log_passk = {
                    f"pass@{k}": passk[k - 1].item()
                    for k in range(1, self.config.device_batch_size + 1)
                }
                wandb.log({"step": step, **log_passk})

            rewards_list = []
            sequence_lengths = []
            reward_tensors = []
            advantage_tensors = []
            loss_values = []
            pg_obj_values = []
            logp_values = []
            valid_token_counts = []
            completion_texts = []
            sample_text_logs = {}
            print(f"Training on {examples_per_step} examples for step {step}...")
            for example_step in range(examples_per_step):
                (
                    sequences_all,
                    generated_texts_all,
                    inputs_all,
                    targets_all,
                    rewards_all,
                    advantages_all,
                ) = next(batch_iterator)
                completion_texts.extend(generated_texts_all)
                reward_tensors.append(rewards_all.detach().cpu())
                advantage_tensors.append(advantages_all.detach().cpu())
                agent.train_mode()
                assert inputs_all.size(0) % self.config.device_batch_size == 0
                num_passes = inputs_all.size(0) // self.config.device_batch_size
                print(
                    f"Example step {example_step} | Generated {inputs_all.size(0)} samples, training in {num_passes} passes"
                )
                for pass_idx in range(num_passes):
                    b0, b1 = (
                        pass_idx * self.config.device_batch_size,
                        (pass_idx + 1) * self.config.device_batch_size,
                    )
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    rewards = rewards_all[b0:b1]
                    advantages = advantages_all[b0:b1]
                    # compute log-probs under autocast (matches scripts/chat_rl.py)
                    with self.autocast_ctx:
                        logp = agent.compute_logp(inputs, targets)
                    pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                    num_valid = (targets >= 0).sum().clamp(min=1)
                    pg_obj = pg_obj / (num_valid * num_passes * examples_per_step)
                    loss = -pg_obj
                    loss.backward()
                    loss_values.append(loss.detach().item())
                    pg_obj_values.append(pg_obj.detach().item())
                    logp_values.append(logp.detach().mean().item())
                    valid_token_counts.append(float(num_valid.item()))
                    print(
                        f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
                    )
                if example_step == 0:
                    preview_count = min(3, len(generated_texts_all))
                    for i in range(preview_count):
                        preview_text = generated_texts_all[i]
                        if len(preview_text) > 512:
                            preview_text = preview_text[:512] + "..."
                        sample_text_logs[f"samples/reward_{i}"] = float(
                            rewards_all[i].item()
                        )
                        sample_text_logs[f"samples/completion_{i}"] = preview_text
                rewards_list.append(rewards_all.mean().item())
                sequence_lengths.extend(len(seq) for seq in sequences_all)

            all_rewards = torch.cat(reward_tensors)
            all_advantages = torch.cat(advantage_tensors)
            has_hash_frac = sum("####" in text for text in completion_texts) / max(
                len(completion_texts), 1
            )
            has_digit_frac = sum(
                any(ch.isdigit() for ch in text) for text in completion_texts
            ) / max(len(completion_texts), 1)
            parsed_frac = sum(
                extract_answer(text) is not None for text in completion_texts
            ) / max(len(completion_texts), 1)
            mean_reward = sum(rewards_list) / len(rewards_list)
            max_reward = max(rewards_list)
            mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
            print(
                f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}"
            )

            lrm = 1.0 - step / num_steps
            for group in agent.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm

            lr_values = [group["lr"] for group in agent.optimizer.param_groups]
            optim_stats = self._compute_optim_stats()

            agent.step_optimizer()
            agent.zero_grad()
            wandb.log(
                {
                    "step": step,
                    "lrm": lrm,
                    "eval/mean_reward": mean_reward,
                    "eval/return_max": max_reward,
                    "eval/sequence_length": mean_sequence_length,
                    "train/reward_mean": all_rewards.mean().item(),
                    "train/reward_std": all_rewards.std(unbiased=False).item(),
                    "train/reward_min": all_rewards.min().item(),
                    "train/reward_max": all_rewards.max().item(),
                    "train/reward_nonzero_frac": (all_rewards != 0)
                    .float()
                    .mean()
                    .item(),
                    "train/format_has_hash_frac": has_hash_frac,
                    "train/format_has_digit_frac": has_digit_frac,
                    "train/format_parsed_frac": parsed_frac,
                    "train/adv_mean": all_advantages.mean().item(),
                    "train/adv_std": all_advantages.std(unbiased=False).item(),
                    "train/adv_abs_mean": all_advantages.abs().mean().item(),
                    "train/adv_nonzero_frac": (all_advantages != 0)
                    .float()
                    .mean()
                    .item(),
                    "train/loss_mean": sum(loss_values) / len(loss_values),
                    "train/loss_min": min(loss_values),
                    "train/loss_max": max(loss_values),
                    "train/pg_obj_mean": sum(pg_obj_values) / len(pg_obj_values),
                    "train/logp_mean": sum(logp_values) / len(logp_values),
                    "train/valid_tokens_mean": sum(valid_token_counts)
                    / len(valid_token_counts),
                    "optim/lr_min": min(lr_values),
                    "optim/lr_max": max(lr_values),
                    "optim/lr_mean": sum(lr_values) / len(lr_values),
                    **sample_text_logs,
                    **optim_stats,
                }
            )

            if (step % self.config.save_every == 0 and step > 0) or (
                step == num_steps - 1
            ):
                model_config_kwargs = dict(agent.model.config.__dict__)
                save_checkpoint(
                    out_dir,
                    step,
                    agent.model.state_dict(),
                    None,  # not saving optimizer state
                    {"model_config": model_config_kwargs},
                )
                print(f"✅ Saved model checkpoint to {out_dir}")
