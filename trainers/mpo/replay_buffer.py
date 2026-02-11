import numpy as np


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self._global_step = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_exec = np.zeros((capacity, act_dim), dtype=np.float32)
        self.actions_raw = np.zeros((capacity, act_dim), dtype=np.float32)
        self.behaviour_logp = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.step_ids = np.zeros((capacity,), dtype=np.int64)

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
        self.step_ids[self.ptr] = self._global_step
        self._global_step += 1
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

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        obs_dim = self.obs.shape[-1]
        act_dim = self.actions_exec.shape[-1]

        obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        next_obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        actions_exec_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        actions_raw_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        rewards_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        dones_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        beh_logp_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)

        filled = 0
        max_tries = batch_size * 200
        tries = 0

        if self.size < self.capacity:
            starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
            for i, start in enumerate(starts):
                idxs = np.arange(start, start + seq_len)
                obs_b[i] = self.obs[idxs]
                next_obs_b[i] = self.next_obs[idxs]
                actions_exec_b[i] = self.actions_exec[idxs]
                actions_raw_b[i] = self.actions_raw[idxs]
                rewards_b[i] = self.rewards[idxs]
                dones_b[i] = self.dones[idxs]
                beh_logp_b[i] = self.behaviour_logp[idxs]
            return {
                "obs": obs_b,
                "actions_exec": actions_exec_b,
                "actions_raw": actions_raw_b,
                "behaviour_logp": beh_logp_b,
                "rewards": rewards_b,
                "next_obs": next_obs_b,
                "dones": dones_b,
            }

        while filled < batch_size and tries < max_tries:
            tries += 1
            start = np.random.randint(0, self.capacity)
            idxs = (start + np.arange(seq_len)) % self.capacity
            step_ids = self.step_ids[idxs]
            if not np.all(step_ids[1:] == step_ids[:-1] + 1):
                continue

            obs_b[filled] = self.obs[idxs]
            next_obs_b[filled] = self.next_obs[idxs]
            actions_exec_b[filled] = self.actions_exec[idxs]
            actions_raw_b[filled] = self.actions_raw[idxs]
            rewards_b[filled] = self.rewards[idxs]
            dones_b[filled] = self.dones[idxs]
            beh_logp_b[filled] = self.behaviour_logp[idxs]
            filled += 1

        if filled < batch_size:
            raise RuntimeError(
                "Failed to sample enough contiguous sequences; consider increasing replay size or reducing seq_len."
            )

        return {
            "obs": obs_b,
            "actions_exec": actions_exec_b,
            "actions_raw": actions_raw_b,
            "behaviour_logp": beh_logp_b,
            "rewards": rewards_b,
            "next_obs": next_obs_b,
            "dones": dones_b,
        }
