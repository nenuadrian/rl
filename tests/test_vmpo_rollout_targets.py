import numpy as np

from trainers.vmpo.targets import compute_rollout_targets


def test_vmpo_returns_targets_match_expected_values():
    rewards = np.array(
        [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    dones = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    values = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    last_value = np.array([0.7, 0.8], dtype=np.float32)

    returns, advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="returns",
        gae_lambda=0.95,
    )

    expected_returns = np.array(
        [[3.88, 1.10], [3.20, -1.00], [3.00, 0.72]],
        dtype=np.float32,
    )
    expected_advantages = expected_returns - values

    np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        advantages, expected_advantages, rtol=1e-6, atol=1e-6
    )


def test_vmpo_dae_targets_match_expected_values():
    rewards = np.array(
        [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    dones = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    values = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    last_value = np.array([0.7, 0.8], dtype=np.float32)

    returns, advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="dae",
        gae_lambda=0.95,
    )

    expected_returns = np.array(
        [[1.27, 2.36], [0.95, -1.00], [3.00, 0.72]],
        dtype=np.float32,
    )
    expected_advantages = expected_returns - values

    np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        advantages, expected_advantages, rtol=1e-6, atol=1e-6
    )


def test_vmpo_gae_lambda_edges_match_expected_estimators():
    rewards = np.array(
        [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    dones = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    values = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )
    last_value = np.array([0.7, 0.8], dtype=np.float32)

    dae_returns, dae_advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="dae",
        gae_lambda=0.95,
    )
    gae0_returns, gae0_advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="gae",
        gae_lambda=0.0,
    )
    mc_returns, mc_advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="returns",
        gae_lambda=0.95,
    )
    gae1_returns, gae1_advantages = compute_rollout_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=0.9,
        estimator="gae",
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(gae0_returns, dae_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gae0_advantages, dae_advantages, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gae1_returns, mc_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gae1_advantages, mc_advantages, rtol=1e-6, atol=1e-6)
