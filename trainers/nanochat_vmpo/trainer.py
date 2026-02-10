import torch
import itertools
import wandb
from nanochat.tasks.gsm8k import GSM8K
from .agent import VMPOAgent

from nanochat.checkpoint_manager import save_checkpoint
import os


class VMPOTrainer:
    def __init__(self, agent: VMPOAgent, args, device):
        self.agent = agent
        self.args = args
        self.device = device
        self.train_task = GSM8K(subset="main", split="train")
        self.val_task = GSM8K(subset="main", split="test")
        self.tokenizer = agent.tokenizer
        self.engine = agent.engine
        self.num_steps = (
            len(self.train_task) // args.examples_per_step
        ) * args.num_epochs

    def get_batch(self):
        args = self.args
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
            num_sampling_steps = args.num_samples // args.device_batch_size
            for sampling_step in range(num_sampling_steps):
                seed = None  # Optionally set as in original script
                batch, batch_masks = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
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
        for idx in range(0, max_examples):
            conversation = val_task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
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
            yield record

    def train(self, out_dir="chatrl_checkpoints"):
        args = self.args
        agent = self.agent
        device = self.device
        num_steps = self.num_steps
        examples_per_rank = args.examples_per_step
        batch_iterator = self.get_batch()
        for step in range(num_steps):
            print(f"=== Starting step {step}/{num_steps} ===")
            if step % args.eval_every == 0:
                agent.eval_mode()
                passk = torch.zeros(args.device_batch_size, device=device)
                records_iter = self.run_gsm8k_eval(
                    max_examples=args.eval_examples,
                    num_samples=args.device_batch_size,
                    temperature=1.0,
                )
                records = list(records_iter)
                for k in range(1, args.device_batch_size + 1):
                    passk[k - 1] = sum(
                        any(o["is_correct"] for o in r["outcomes"][:k]) for r in records
                    )
                num_records = torch.tensor(
                    len(records), dtype=torch.long, device=device
                )
                passk = passk / num_records.item()
                print_passk = [
                    f"Pass@{k}: {passk[k - 1].item():.4f}"
                    for k in range(1, args.device_batch_size + 1)
                ]
                print(f"Step {step} | {', '.join(print_passk)}")
                log_passk = {
                    f"pass@{k}": passk[k - 1].item()
                    for k in range(1, args.device_batch_size + 1)
                }
                wandb.log({"step": step, **log_passk})

            rewards_list = []
            sequence_lengths = []
            for example_step in range(examples_per_rank):
                sequences_all, inputs_all, targets_all, rewards_all, advantages_all = (
                    next(batch_iterator)
                )
                agent.train_mode()
                assert inputs_all.size(0) % args.device_batch_size == 0
                num_passes = inputs_all.size(0) // args.device_batch_size
                for pass_idx in range(num_passes):
                    b0, b1 = (
                        pass_idx * args.device_batch_size,
                        (pass_idx + 1) * args.device_batch_size,
                    )
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    rewards = rewards_all[b0:b1]
                    advantages = advantages_all[b0:b1]
                    logp = agent.compute_logp(inputs, targets)
                    # --- VMPO: Compute KLs and update duals (placeholders) ---
                    # Compute KLs between current and reference model
                    kl_mu, kl_sigma = agent.compute_kl(inputs, targets)

                    # Update dual variables (eta, alpha_mu, alpha_sigma)
                    agent.update_duals(advantages, kl_mu, kl_sigma)

                    # --- Policy loss (VMPO-style) ---
                    weights = agent.compute_importance_weights(advantages)
                    weighted_nll = -(weights.detach() * logp).sum()
                    num_valid = (targets >= 0).sum().clamp(min=1)
                    weighted_nll = weighted_nll / (
                        num_valid * num_passes * examples_per_rank
                    )
                    # Add KL penalties (now real KLs)
                    alpha_mu = agent.log_alpha_mu.exp().detach()
                    alpha_sigma = agent.log_alpha_sigma.exp().detach()
                    policy_loss = (
                        weighted_nll + alpha_mu * kl_mu + alpha_sigma * kl_sigma
                    )
                    policy_loss.backward()
                    print(
                        f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {policy_loss.item():.6f} | KL: {kl_mu.item():.4f}±{kl_sigma.item():.4f} | Average reward: {rewards.mean().item()}"
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

            # Save checkpoint every save_every steps, and at last step
            if (step % args.save_every == 0 and step > 0) or (step == num_steps - 1):
                # Use same logic as chat_rl_single_gpu.py for checkpoint dir/tag
                depth = agent.model.config.n_layer
                output_dirname = args.model_tag if args.model_tag else f"d{depth}"
                checkpoint_dir = os.path.join(out_dir, output_dirname)
                model_config_kwargs = dict(agent.model.config.__dict__)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    agent.model.state_dict(),
                    None,  # not saving optimizer state
                    {"model_config": model_config_kwargs},
                )
                print(f"✅ Saved model checkpoint to {checkpoint_dir}")
