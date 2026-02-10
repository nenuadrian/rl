# Agent for chat RL (GRPO/REINFORCE style)
import torch
from nanochat.engine import Engine
from nanochat.checkpoint_manager import build_model, find_last_step

from dataclasses import dataclass


@dataclass
class ChatRLConfig:
    num_epochs: int = 1
    device_batch_size: int = 8
    examples_per_step: int = 16
    num_samples: int = 16
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    init_lr_frac: float = 0.05
    eval_every: int = 60
    eval_examples: int = 400
    save_every: int = 60
    model_step: int | None = None
    dtype: str = "bfloat16"


class ChatRLAgent:
    def __init__(
        self,
        device,
        checkpoint_dir,
        embedding_lr,
        unembedding_lr,
        matrix_lr,
        weight_decay,
        init_lr_frac,
        model_step=None,
        dtype="bfloat16",
    ):
        self.device = device
        self.ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
        if model_step is None:
            # guess the step by defaulting to the last step
            model_step = find_last_step(checkpoint_dir)
        self.model, self.tokenizer, self.meta = build_model(
            checkpoint_dir, model_step, device, "rl"
        )
        self.engine = Engine(self.model, self.tokenizer)
        self.optimizer = self.model.setup_optimizer(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]

    def act(self, tokens, num_samples, max_tokens, temperature, top_k, seed=None):
        # Returns generated_token_sequences, masks from engine.generate_batch
        return self.engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )

    def compute_logp(self, inputs, targets, loss_reduction="none"):
        # Returns negative log-probabilities for REINFORCE objective
        return -self.model(inputs, targets, loss_reduction=loss_reduction).view_as(
            inputs
        )

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step_optimizer(self):
        self.optimizer.step()
