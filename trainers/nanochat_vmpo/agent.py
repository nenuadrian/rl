# Agent for chat RL (GRPO/REINFORCE style)
import torch
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

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
    model_tag: str | None = None
    model_step: int | None = None
    dtype: str = "bfloat16"


class VMPOAgent:
    def __init__(self, device, model_tag=None, model_step=None, dtype="bfloat16", config=None, reference_model_tag=None, reference_model_step=None):
        self.device = device
        self.ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self.model, self.tokenizer, self.meta = load_model(
            "base", device, phase="eval", model_tag=model_tag, step=model_step
        )
        self.engine = Engine(self.model, self.tokenizer)
        self.optimizer = None
        self.config = config

        # Reference model for KL constraint (frozen)
        if reference_model_tag is not None or reference_model_step is not None:
            self.reference_model, _, _ = load_model(
                "base", device, phase="eval", model_tag=reference_model_tag, step=reference_model_step
            )
            self.reference_model.eval()
            for p in self.reference_model.parameters():
                p.requires_grad = False
        else:
            self.reference_model = None

        # VMPO dual variables (eta, alpha_mu, alpha_sigma)
        self.log_temperature = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.log_alpha_mu = torch.nn.Parameter(torch.zeros(1, device=device))
        self.log_alpha_sigma = torch.nn.Parameter(torch.zeros(1, device=device))
        self.eta_opt = torch.optim.Adam([self.log_temperature], lr=getattr(config, "temperature_lr", 1e-3)) if config else None
        self.alpha_opt = torch.optim.Adam([self.log_alpha_mu, self.log_alpha_sigma], lr=getattr(config, "alpha_lr", 1e-3)) if config else None

    def compute_kl(self, inputs, targets):
        """
        Compute mean and std KL between current and reference model logits (per token, masked).
        Returns: kl_mean (scalar), kl_std (scalar)
        """
        if self.reference_model is None:
            return torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)
        with torch.no_grad():
            # Get logits from both models
            logits_current = self.model(inputs)  # (B, T, V)
            logits_ref = self.reference_model(inputs)  # (B, T, V)
            # Mask for valid tokens
            mask = (targets >= 0)
            # Softmax probabilities
            p = torch.nn.functional.log_softmax(logits_current, dim=-1)
            q = torch.nn.functional.log_softmax(logits_ref, dim=-1)
            # KL per token: sum p * (log p - log q)
            p_probs = p.exp()
            kl = (p_probs * (p - q)).sum(dim=-1)  # (B, T)
            kl = kl[mask]
            if kl.numel() == 0:
                return torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)
            return kl.mean(), kl.std()

    def setup_optimizer(self, embedding_lr, unembedding_lr, matrix_lr, weight_decay, init_lr_frac):
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
        return -self.model(inputs, targets, loss_reduction=loss_reduction).view_as(inputs)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step_optimizer(self):
        self.optimizer.step()

    # VMPO dual variable update logic
    def update_duals(self, advantages, kl_mu, kl_sigma):
        # advantages: (batch,) tensor
        # kl_mu, kl_sigma: scalars (mean KL for mean and stddev)
        if self.config is None or self.eta_opt is None or self.alpha_opt is None:
            return

        # E-step: update temperature (eta)
        adv = advantages.detach().flatten()
        k = max(1, int(self.config.topk_fraction * adv.numel()))
        topk_vals, _ = torch.topk(adv, k)
        threshold = topk_vals.min()
        mask_bool = adv >= threshold
        A = adv[mask_bool]
        A = A - A.mean()  # center only
        K = A.numel()

        temperature = torch.exp(self.log_temperature) + 1e-8
        logK = torch.log(torch.tensor(float(K), device=A.device))
        dual_loss = temperature * self.config.epsilon_eta + temperature * (
            torch.logsumexp(A / temperature, dim=0) - logK
        )

        self.eta_opt.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.eta_opt.step()

        # M-step: update KL duals (alpha_mu, alpha_sigma)
        alpha_mu = self.log_alpha_mu.exp()
        alpha_sigma = self.log_alpha_sigma.exp()
        alpha_loss = alpha_mu * (self.config.epsilon_mu - kl_mu.detach()) \
            + alpha_sigma * (self.config.epsilon_sigma - kl_sigma.detach())

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

    def compute_importance_weights(self, advantages):
        # Compute importance weights for VMPO policy loss
        if self.config is None:
            return torch.ones_like(advantages)
        temperature = torch.exp(self.log_temperature) + 1e-8
        adv = advantages.detach().flatten()
        k = max(1, int(self.config.topk_fraction * adv.numel()))
        topk_vals, _ = torch.topk(adv, k)
        threshold = topk_vals.min()
        mask_bool = adv >= threshold
        A = adv[mask_bool]
        A = A - A.mean()
        weights = torch.zeros_like(adv)
        weights[mask_bool] = torch.softmax(A / temperature, dim=0)
        return weights.reshape(advantages.shape)

