from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ObsNormalizer:
    """Running mean/variance normalizer for flat observations.

    Supports:
      - single obs: (obs_dim,)
      - batched obs: (N, obs_dim)
    """

    obs_dim: int
    eps: float = 1e-8
    clip: float | None = 10.0

    def __post_init__(self) -> None:
        self._count: float = 0.0
        self._mean = np.zeros((self.obs_dim,), dtype=np.float64)
        self._m2 = np.zeros((self.obs_dim,), dtype=np.float64)

    def update(self, obs: np.ndarray) -> None:
        x = np.asarray(obs, dtype=np.float64)

        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected obs with ndim 1 or 2, got shape {x.shape}")

        if x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {x.shape[1]}")

        # Welford update, one sample at a time
        for i in range(x.shape[0]):
            self._count += 1.0
            delta = x[i] - self._mean
            self._mean += delta / self._count
            delta2 = x[i] - self._mean
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
        x = np.asarray(obs, dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        elif x.ndim == 2:
            squeeze = False
        else:
            raise ValueError(f"Expected obs with ndim 1 or 2, got shape {x.shape}")

        if x.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {x.shape[1]}")

        x = (x - self.mean) / np.sqrt(self.var + self.eps)

        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)

        x = x.astype(np.float32, copy=False)
        return x[0] if squeeze else x
