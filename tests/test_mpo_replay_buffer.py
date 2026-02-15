import numpy as np
import torch

from trainers.mpo.replay_buffer import MPOReplayBuffer


def _add_transition(buf: MPOReplayBuffer, step: int, obs_dim: int, act_dim: int) -> None:
    obs = np.full((obs_dim,), float(step), dtype=np.float32)
    action_exec = np.full((act_dim,), float(step), dtype=np.float32)
    action_raw = np.full((act_dim,), float(step) + 0.5, dtype=np.float32)
    reward = float(step)
    next_obs = obs + 1.0
    done = float(step % 2)
    buf.add(obs, action_exec, action_raw, float(step), reward, next_obs, done)


def test_mpo_replay_buffer_sample_shapes():
    obs_dim = 4
    act_dim = 2
    capacity = 16
    buf = MPOReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=capacity)

    for step in range(10):
        _add_transition(buf, step, obs_dim, act_dim)

    batch = buf.sample(batch_size=6)
    assert buf.size == 10
    assert buf.capacity == capacity
    assert isinstance(batch["obs"], torch.Tensor)
    assert batch["obs"].shape == (6, obs_dim)
    assert batch["actions"].shape == (6, act_dim)
    assert batch["rewards"].shape == (6, 1)
    assert batch["next_obs"].shape == (6, obs_dim)
    assert batch["dones"].shape == (6, 1)


def test_mpo_replay_buffer_sequence_sampling_avoids_ring_wrap():
    obs_dim = 3
    act_dim = 2
    capacity = 5
    seq_len = 4
    buf = MPOReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=capacity)

    for step in range(6):
        _add_transition(buf, step, obs_dim, act_dim)

    batch = buf.sample_sequences(batch_size=64, seq_len=seq_len)
    obs = batch["obs"]
    assert obs.shape == (64, seq_len, obs_dim)

    first_obs_dim = obs[..., 0]
    diffs = first_obs_dim[:, 1:] - first_obs_dim[:, :-1]
    assert torch.all(diffs == 1.0)
