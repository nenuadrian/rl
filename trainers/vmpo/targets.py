from __future__ import annotations

from typing import Literal

import numpy as np


def compute_returns_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)

    T, N = rewards_np.shape
    returns = np.zeros_like(rewards_np, dtype=np.float32)

    running_return = np.zeros(N, dtype=np.float32)
    last_val_arr = np.asarray(last_value, dtype=np.float32)
    if last_val_arr.ndim == 0:
        running_return[:] = float(last_val_arr)
    else:
        running_return[:] = last_val_arr

    for t in reversed(range(T)):
        running_return = rewards_np[t] + gamma * (1.0 - dones_np[t]) * running_return
        returns[t] = running_return

    if returns.shape[1] == 1:
        return returns.reshape(-1)
    return returns


def compute_dae_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)

    T, N = rewards_np.shape
    last_val_arr = np.asarray(last_value, dtype=np.float32)
    if last_val_arr.ndim == 0:
        last_val_arr = np.full((N,), float(last_val_arr), dtype=np.float32)
    else:
        last_val_arr = last_val_arr.reshape(N)

    next_values = np.zeros((T, N), dtype=np.float32)
    if T > 1:
        next_values[:-1] = values_np[1:]
    next_values[-1] = last_val_arr

    returns = rewards_np + gamma * (1.0 - dones_np) * next_values
    advantages = returns - values_np

    if returns.shape[1] == 1:
        return returns.reshape(-1), advantages.reshape(-1)
    return returns, advantages


def compute_gae_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)

    T, N = rewards_np.shape
    last_val_arr = np.asarray(last_value, dtype=np.float32)
    if last_val_arr.ndim == 0:
        last_val_arr = np.full((N,), float(last_val_arr), dtype=np.float32)
    else:
        last_val_arr = last_val_arr.reshape(N)

    next_values = np.zeros((T, N), dtype=np.float32)
    if T > 1:
        next_values[:-1] = values_np[1:]
    next_values[-1] = last_val_arr

    deltas = rewards_np + gamma * (1.0 - dones_np) * next_values - values_np
    advantages = np.zeros_like(rewards_np, dtype=np.float32)
    lastgaelam = np.zeros((N,), dtype=np.float32)

    for t in reversed(range(T)):
        lastgaelam = deltas[t] + gamma * gae_lambda * (1.0 - dones_np[t]) * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values_np

    if returns.shape[1] == 1:
        return returns.reshape(-1), advantages.reshape(-1)
    return returns, advantages


def compute_rollout_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    estimator: Literal["returns", "dae", "gae"],
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if estimator == "returns":
        returns = np.asarray(
            compute_returns_targets(rewards, dones, last_value, gamma),
            dtype=np.float32,
        )
        values_np = np.asarray(values, dtype=np.float32)
        if values_np.ndim == 1:
            values_np = values_np.reshape(-1, 1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        advantages = returns - values_np
        return returns, advantages
    if estimator == "dae":
        return compute_dae_targets(rewards, dones, values, last_value, gamma)
    if estimator == "gae":
        return compute_gae_targets(
            rewards,
            dones,
            values,
            last_value,
            gamma,
            gae_lambda,
        )
    raise ValueError(f"Unknown advantage estimator: {estimator}")
