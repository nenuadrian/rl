import torch
import itertools
import wandb
from contextlib import nullcontext
from nanochat.tasks.gsm8k import GSM8K
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

    def get_batch(self):
        train_task = self.train_task
        tokenizer = self.tokenizer
        device = self.device
        engine = self.engine
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(0, len(train_task))
        for example_idx in itertools.cycle(rank_indices):
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            generated_token_sequences = []
            masks = []
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
            yield generated_token_sequences, inputs, targets, rewards, advantages

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
        print(f"Running GSM8K evaluation on {max_examples} examples with {num_samples} samples each...")
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

    def train(self, out_dir: str):
        agent = self.agent
        device = self.device
        num_steps = self.num_steps
        examples_per_rank = self.config.examples_per_step
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
            print(f"Training on {examples_per_rank} examples for step {step}...")
            for example_step in range(examples_per_rank):
                sequences_all, inputs_all, targets_all, rewards_all, advantages_all = (
                    next(batch_iterator)
                )
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
                    pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
                    loss = -pg_obj
                    loss.backward()
                    print(
                        f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
                    )
                rewards_list.append(rewards_all.mean().item())
                sequence_lengths.extend(len(seq) for seq in sequences_all)

            mean_reward = sum(rewards_list) / len(rewards_list)
            mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
            print(
                f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}"
            )
            wandb.log(
                {
                    "step": step,
                    "reward": mean_reward,
                    "sequence_length": mean_sequence_length,
                }
            )

            lrm = 1.0 - step / num_steps
            for group in agent.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
            agent.step_optimizer()
            agent.zero_grad()
            wandb.log({"step": step, "lrm": lrm})

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
