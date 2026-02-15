import numpy as np


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_exec = np.zeros((capacity, act_dim), dtype=np.float32)
        self.actions_raw = np.zeros((capacity, act_dim), dtype=np.float32)
        self.behaviour_logp = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._sequence_scratch = None
        self._sequence_scratch_shape = None
        self._sequence_offsets = {}

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
        self.obs[self.ptr] = obs
        self.actions_exec[self.ptr] = action_exec
        self.actions_raw[self.ptr] = action_raw
        self.behaviour_logp[self.ptr] = behaviour_logp
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions_exec[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }

    def _get_sequence_offsets(self, seq_len: int) -> np.ndarray:
        offsets = self._sequence_offsets.get(seq_len)
        if offsets is None:
            offsets = np.arange(seq_len, dtype=np.int64)
            self._sequence_offsets[seq_len] = offsets
        return offsets

    def _get_sequence_scratch(self, batch_size: int, seq_len: int) -> dict:
        shape = (batch_size, seq_len)
        if self._sequence_scratch_shape != shape:
            obs_dim = self.obs.shape[-1]
            act_dim = self.actions_exec.shape[-1]
            self._sequence_scratch = {
                "obs": np.empty((batch_size, seq_len, obs_dim), dtype=np.float32),
                "next_obs": np.empty((batch_size, seq_len, obs_dim), dtype=np.float32),
                "actions_exec": np.empty(
                    (batch_size, seq_len, act_dim), dtype=np.float32
                ),
                "actions_raw": np.empty(
                    (batch_size, seq_len, act_dim), dtype=np.float32
                ),
                "rewards": np.empty((batch_size, seq_len, 1), dtype=np.float32),
                "dones": np.empty((batch_size, seq_len, 1), dtype=np.float32),
                "behaviour_logp": np.empty((batch_size, seq_len, 1), dtype=np.float32),
            }
            self._sequence_scratch_shape = shape
        return self._sequence_scratch

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        scratch = self._get_sequence_scratch(batch_size, seq_len)
        offsets = self._get_sequence_offsets(seq_len)
        if self.size < self.capacity:
            starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
            idxs = starts[:, None] + offsets[None, :]
        else:
            # Once the ring is full there is a single discontinuity at ptr-1 -> ptr.
            # Valid sequence starts are the contiguous block [ptr, ptr + valid_count).
            valid_start_count = self.capacity - (seq_len - 1)
            if valid_start_count <= 0:
                raise RuntimeError(
                    "Failed to sample enough contiguous sequences; consider reducing seq_len."
                )
            starts = (
                self.ptr + np.random.randint(0, valid_start_count, size=batch_size)
            ) % self.capacity
            idxs = (starts[:, None] + offsets[None, :]) % self.capacity

        scratch["obs"][:] = self.obs[idxs]
        scratch["next_obs"][:] = self.next_obs[idxs]
        scratch["actions_exec"][:] = self.actions_exec[idxs]
        scratch["actions_raw"][:] = self.actions_raw[idxs]
        scratch["rewards"][:] = self.rewards[idxs]
        scratch["dones"][:] = self.dones[idxs]
        scratch["behaviour_logp"][:] = self.behaviour_logp[idxs]

        return {
            "obs": scratch["obs"],
            "actions_exec": scratch["actions_exec"],
            "actions_raw": scratch["actions_raw"],
            "behaviour_logp": scratch["behaviour_logp"],
            "rewards": scratch["rewards"],
            "next_obs": scratch["next_obs"],
            "dones": scratch["dones"],
        }
