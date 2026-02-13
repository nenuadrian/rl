import numpy as np
import pytest
import torch

from pathlib import Path

from hyperparameters.ppo_lm import get as get_lm_preset
from hyperparameters.vmpo import get as get_vmpo_preset
from hyperparameters.vmpo_sgd import get as get_vmpo_sgd_preset
from trainers.ppo_lm.trainer import LMGRPOConfig, PPOLMTrainer
from trainers.ppo.agent import PPOAgent
from trainers.trpo.agent import TRPOAgent, TRPOConfig
from trainers.vmpo.agent import VMPOAgent, VMPOConfig
from trainers.mpo.agent import MPOAgent, MPOConfig


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
        policy_layer_sizes=(32, 32),
        critic_layer_sizes=(32, 32),
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
        policy_layer_sizes=(32, 32),
        config=VMPOConfig(
            gamma=0.99,
            policy_lr=3e-4,
            value_lr=1e-3,
            topk_fraction=0.2,
            temperature_init=0.1,
            temperature_lr=1e-3,
            epsilon_eta=0.1,
            epsilon_mu=0.1,
            epsilon_sigma=0.1,
            alpha_lr=1e-3,
            max_grad_norm=0.5,
        ),
    )
    assert isinstance(agent.opt, torch.optim.Adam)
    assert isinstance(agent.eta_opt, torch.optim.Adam)
    assert isinstance(agent.alpha_opt, torch.optim.Adam)

    obs = np.random.randn(obs_dim).astype(np.float32)
    action, value, mean, log_std = agent.act(obs, deterministic=False)

    assert action.shape == (act_dim,)
    assert mean.shape == (act_dim,)
    assert log_std.shape == (act_dim,)
    assert isinstance(value, float)

    batch_size = 12
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)
    returns = torch.randn(batch_size, 1)
    advantages = torch.randn(batch_size, 1)
    old_means = torch.randn(batch_size, act_dim)
    old_log_stds = torch.randn(batch_size, act_dim)

    batch = {
        "obs": obs,
        "actions": actions,
        "returns": returns,
        "advantages": advantages,
        "old_means": old_means,
        "old_log_stds": old_log_stds,
    }

    metrics = agent.update(batch)
    assert "loss/policy" in metrics
    assert "kl/mean" in metrics
    assert "kl/std" in metrics


def test_vmpo_agent_supports_sgd_optimizer():
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
        policy_layer_sizes=(32, 32),
        config=VMPOConfig(optimizer_type="sgd"),
    )

    assert isinstance(agent.opt, torch.optim.SGD)
    assert isinstance(agent.eta_opt, torch.optim.SGD)
    assert isinstance(agent.alpha_opt, torch.optim.SGD)


def test_vmpo_preset_optimizer_selection():
    env_id = "HalfCheetah-v5"
    vmpo_preset = get_vmpo_preset(env_id)
    vmpo_sgd_preset = get_vmpo_sgd_preset(env_id)

    assert vmpo_preset["optimizer_type"] == "adam"
    assert vmpo_sgd_preset["optimizer_type"] == "sgd"


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
        policy_layer_sizes=(32, 32),
        critic_layer_sizes=(32, 32),
        config=MPOConfig(),
    )

    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.act(obs, deterministic=False)
    assert action.shape == (act_dim,)
    assert np.all(action >= action_low - 1e-5)
    assert np.all(action <= action_high + 1e-5)

    action_det1 = agent.act(obs, deterministic=True)
    action_det2 = agent.act(obs, deterministic=True)
    assert action_det1.shape == (act_dim,)
    assert np.allclose(action_det1, action_det2)

    batch_size = 10
    batch = {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "actions": np.random.randn(batch_size, act_dim).astype(np.float32),
        "rewards": np.random.randn(batch_size, 1).astype(np.float32),
        "next_obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "dones": np.zeros((batch_size, 1), dtype=np.float32),
    }

    def _clone_params(module: torch.nn.Module):
        return [p.detach().clone() for p in module.parameters()]

    def _any_param_changed(before, module: torch.nn.Module) -> bool:
        for before_tensor, param in zip(before, module.parameters()):
            if not torch.allclose(before_tensor, param.detach()):
                return True
        return False

    q_before = _clone_params(agent.q)
    policy_before = _clone_params(agent.policy)

    metrics = agent.update(batch)

    expected_keys = {
        "loss/q",
        "loss/policy",
        "loss/dual_eta",
        "loss/dual",
        "kl/q_pi",
        "kl/mean",
        "kl/std",
        "eta",
        "lambda",
    }
    assert expected_keys.issubset(metrics.keys())

    # Basic metric sanity: scalars and finite.
    for key in expected_keys:
        value = metrics[key]
        assert isinstance(value, (int, float, np.floating))
        assert np.isfinite(float(value))

    assert metrics["eta"] > 0.0
    assert metrics["lambda"] > 0.0

    # Update should actually modify parameters.
    assert _any_param_changed(q_before, agent.q)
    assert _any_param_changed(policy_before, agent.policy)


def test_trpo_agent_act_and_update():
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 5
    act_dim = 2
    action_low = -np.ones(act_dim, dtype=np.float32)
    action_high = np.ones(act_dim, dtype=np.float32)

    agent = TRPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
        policy_layer_sizes=(32, 32),
        critic_layer_sizes=(32, 32),
        config=TRPOConfig(
            target_kl=0.02,
            cg_iters=5,
            backtrack_iters=5,
            value_epochs=2,
            value_minibatch_size=16,
        ),
    )

    obs = torch.randn(12, obs_dim)
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
    expected_keys = {
        "policy/surrogate_before",
        "policy/surrogate_after",
        "policy/kl",
        "policy/line_search_accepted",
        "loss/value",
        "value/mae",
    }
    assert expected_keys.issubset(metrics.keys())
