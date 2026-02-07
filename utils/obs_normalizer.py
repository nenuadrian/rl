from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObsNormalizer:
    """Running mean/variance normalizer for flat float32 observations."""

    obs_dim: int
    eps: float = 1e-8
    clip: float | None = 10.0

    def __post_init__(self) -> None:
        self._count: float = 0.0
        self._mean = np.zeros((self.obs_dim,), dtype=np.float64)
        self._m2 = np.zeros((self.obs_dim,), dtype=np.float64)

    def update(self, obs: np.ndarray) -> None:
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {x.shape[0]}")

        self._count += 1.0
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        return self._mean.astype(np.float32, copy=False)

    @property
    def var(self) -> np.ndarray:
        if self._count < 2:
            return np.ones((self.obs_dim,), dtype=np.float32)
        return (self._m2 / (self._count - 1.0)).astype(np.float32, copy=False)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        v = self.var
        x = (x - self.mean) / np.sqrt(v + self.eps)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x.astype(np.float32, copy=False)
