from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """In-network running observation normalizer.

    Tracks running mean/variance via parallel Welford updates stored as
    non-parameter buffers (included in ``state_dict``, no gradients).

    * ``model.train()`` — running stats update on every forward call.
    * ``model.eval()``  — stats are frozen; forward only normalizes.

    All computation stays on-device in float32.
    """

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.clip = clip
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x)
        return self._normalize(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        batch = x.detach()
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)

        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (
            m_a + m_b + delta.square() * self.count * batch_count / total_count
        ) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(
            -self.clip, self.clip
        )
