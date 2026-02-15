from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")

        self._buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.capacity),
        )
        self._sequence_offsets: dict[int, torch.Tensor] = {}

    @property
    def size(self) -> int:
        return int(len(self._buffer))

    @property
    def ptr(self) -> int:
        return int(self._buffer.write_count % self.capacity)

    @staticmethod
    def _as_tensor(value: np.ndarray | float, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device="cpu").reshape(shape)

    def add(
        self,
        obs: np.ndarray,
        action_exec: np.ndarray,
        action_raw: np.ndarray,
        behaviour_logp: float,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        transition = TensorDict(
            {
                "obs": self._as_tensor(obs, (self.obs_dim,)),
                "next_obs": self._as_tensor(next_obs, (self.obs_dim,)),
                "actions_exec": self._as_tensor(action_exec, (self.act_dim,)),
                "actions_raw": self._as_tensor(action_raw, (self.act_dim,)),
                "behaviour_logp": self._as_tensor(behaviour_logp, (1,)),
                "rewards": self._as_tensor(reward, (1,)),
                "dones": self._as_tensor(done, (1,)),
            },
            batch_size=[],
        )
        self._buffer.add(transition)

    def sample(self, batch_size: int) -> dict:
        sampled = self._buffer.sample(batch_size=int(batch_size))
        return {
            "obs": sampled["obs"],
            "actions": sampled["actions_exec"],
            "rewards": sampled["rewards"],
            "next_obs": sampled["next_obs"],
            "dones": sampled["dones"],
        }

    def _get_sequence_offsets(self, seq_len: int) -> torch.Tensor:
        offsets = self._sequence_offsets.get(seq_len)
        if offsets is None:
            offsets = torch.arange(seq_len, dtype=torch.long)
            self._sequence_offsets[seq_len] = offsets
        return offsets

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        batch_size = int(batch_size)
        offsets = self._get_sequence_offsets(seq_len)
        if self.size < self.capacity:
            starts = torch.randint(
                0,
                self.size - seq_len + 1,
                (batch_size,),
                dtype=torch.long,
            )
            idxs = starts.unsqueeze(1) + offsets.unsqueeze(0)
        else:
            valid_start_count = self.capacity - (seq_len - 1)
            if valid_start_count <= 0:
                raise RuntimeError(
                    "Failed to sample enough contiguous sequences; consider reducing seq_len."
                )
            starts = (
                self.ptr
                + torch.randint(0, valid_start_count, (batch_size,), dtype=torch.long)
            ) % self.capacity
            idxs = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.capacity

        sampled = self._buffer.storage[idxs]
        return {
            "obs": sampled["obs"],
            "actions_exec": sampled["actions_exec"],
            "actions_raw": sampled["actions_raw"],
            "behaviour_logp": sampled["behaviour_logp"],
            "rewards": sampled["rewards"],
            "next_obs": sampled["next_obs"],
            "dones": sampled["dones"],
        }
