from dataclasses import dataclass
from typing import Iterator
import numpy as np

# =========================
# Rollout buffer (time-major, vectorised-safe)
# =========================


@dataclass
class RolloutBuffer:
    obs: np.ndarray  # (T, N, obs_dim)
    actions: np.ndarray  # (T, N, act_dim)
    rewards: np.ndarray  # (T, N)
    dones: np.ndarray  # (T, N)
    values: np.ndarray  # (T, N)
    log_probs: np.ndarray  # (T, N)
    T: int
    N: int
    ptr: int = 0

    @classmethod
    def create(
        cls, obs_dim: int, act_dim: int, rollout_steps: int, num_envs: int
    ) -> "RolloutBuffer":
        return cls(
            obs=np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32),
            actions=np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32),
            rewards=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            dones=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            values=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            log_probs=np.zeros((rollout_steps, num_envs), dtype=np.float32),
            T=rollout_steps,
            N=num_envs,
            ptr=0,
        )

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        t: int,
        env_i: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs[t, env_i] = obs
        self.actions[t, env_i] = action
        self.rewards[t, env_i] = reward
        self.dones[t, env_i] = done
        self.values[t, env_i] = value
        self.log_probs[t, env_i] = log_prob
        self.ptr = max(self.ptr, t + 1)

    def compute_returns_advantages(
        self,
        last_values: np.ndarray,  # shape (N,)
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, N = self.ptr, self.N
        if T <= 0:
            raise ValueError("RolloutBuffer is empty.")

        obs = self.obs[:T]
        actions = self.actions[:T]
        rewards = self.rewards[:T]
        dones = self.dones[:T]
        values = self.values[:T]
        log_probs = self.log_probs[:T]

        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]

            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # flatten (T, N) -> (T*N)
        return (
            obs.reshape(T * N, -1),
            actions.reshape(T * N, -1),
            log_probs.reshape(T * N),
            values.reshape(T * N),
            returns.reshape(T * N),
            advantages.reshape(T * N),
        )

    def minibatches(self, batch_size: int) -> Iterator[np.ndarray]:
        idx = np.arange(self.ptr * self.N)
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]
