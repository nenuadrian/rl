import numpy as np
import torch

from trainers.sac.agent import SACAgent
from trainers.ppo.agent import PPOAgent
from trainers.vmpo.agent import VMPOAgent
from trainers.mpo.agent import MPOAgent


def test_sac_agent_act_and_update():
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 5
    act_dim = 3
    action_low = -np.ones(act_dim, dtype=np.float32)
    action_high = np.ones(act_dim, dtype=np.float32)

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
        hidden_sizes=(32, 32),
    )

    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.act(obs, deterministic=False)
    assert action.shape == (act_dim,)

    batch_size = 8
    batch = {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "actions": np.random.randn(batch_size, act_dim).astype(np.float32),
        "rewards": np.random.randn(batch_size, 1).astype(np.float32),
        "next_obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "dones": np.zeros((batch_size, 1), dtype=np.float32),
    }

    metrics = agent.update(batch)
    assert "loss/q1" in metrics
    assert "loss/q2" in metrics
    assert "loss/policy" in metrics
    assert "loss/alpha" in metrics
    assert "alpha" in metrics


def test_ppo_agent_act_and_update():
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 4
    act_dim = 2
    action_low = -np.ones(act_dim, dtype=np.float32)
    action_high = np.ones(act_dim, dtype=np.float32)

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
        hidden_sizes=(32, 32),
    )

    batch_size = 16
    obs = torch.randn(batch_size, obs_dim)
    actions, log_probs, values = agent.act(obs, deterministic=False)

    returns = values + torch.randn_like(values) * 0.1
    advantages = torch.randn_like(values)

    batch = {
        "obs": obs,
        "actions": actions.detach(),
        "log_probs": log_probs.detach(),
        "returns": returns.detach(),
        "advantages": advantages.detach(),
    }

    metrics = agent.update(batch)
    assert "loss/policy" in metrics
    assert "loss/value" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics


def test_vmpo_agent_act_and_update():
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 6
    act_dim = 2
    action_low = -np.ones(act_dim, dtype=np.float32)
    action_high = np.ones(act_dim, dtype=np.float32)

    agent = VMPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
        hidden_sizes=(32, 32),
    )

    hidden = agent.init_hidden()
    obs = np.random.randn(obs_dim).astype(np.float32)
    prev_action = np.zeros(act_dim, dtype=np.float32)
    prev_reward = 0.0
    action, value, mean, log_std, hidden = agent.act(
        obs, prev_action, prev_reward, hidden, deterministic=False
    )

    assert action.shape == (act_dim,)
    assert mean.shape == (act_dim,)
    assert log_std.shape == (act_dim,)
    assert isinstance(value, float)

    batch_size = 12
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    prev_actions = torch.zeros_like(actions)
    prev_rewards = torch.zeros(batch_size, 1)
    dones = torch.zeros(batch_size, 1)
    returns = torch.randn(batch_size, 1)
    advantages = torch.randn(batch_size, 1)
    old_means = torch.randn(batch_size, act_dim)
    old_log_stds = torch.randn(batch_size, act_dim)

    batch = {
        "obs": obs,
        "actions": actions,
        "prev_actions": prev_actions,
        "prev_rewards": prev_rewards,
        "dones": dones,
        "returns": returns,
        "advantages": advantages,
        "old_means": old_means,
        "old_log_stds": old_log_stds,
    }

    metrics = agent.update(batch)
    assert "loss/policy" in metrics
    assert "loss/value" in metrics
    assert "kl/mean" in metrics
    assert "kl/std" in metrics


def test_mpo_agent_act_and_update():
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 7
    act_dim = 3
    action_low = -np.ones(act_dim, dtype=np.float32)
    action_high = np.ones(act_dim, dtype=np.float32)

    agent = MPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
        hidden_sizes=(32, 32),
    )

    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.act(obs, deterministic=False)
    assert action.shape == (act_dim,)

    batch_size = 10
    batch = {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "actions": np.random.randn(batch_size, act_dim).astype(np.float32),
        "rewards": np.random.randn(batch_size, 1).astype(np.float32),
        "next_obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "dones": np.zeros((batch_size, 1), dtype=np.float32),
    }

    metrics = agent.update(batch)
    assert "loss/q1" in metrics
    assert "loss/q2" in metrics
    assert "loss/policy" in metrics
    assert "loss/dual_eta" in metrics
    assert "kl/q_pi" in metrics
    assert "eta" in metrics
    assert "lambda" in metrics
