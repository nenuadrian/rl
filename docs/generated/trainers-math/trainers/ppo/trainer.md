# `trainers.ppo.trainer` Math-Annotated Source

_Source: `minerva/trainers/ppo/trainer.py`_

Each `# LaTeX:` annotation is rendered with its source line and 10 following lines of context.

## Rendered Math Annotations

### Line 264

```python
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            # LaTeX: \log \pi_{\theta}(a_t|s_t) = \sum_j \log \mathcal{N}(a_{t,j}; \mu_{t,j}, \sigma_{t,j})
            probs.log_prob(action).sum(1),
            # LaTeX: \mathcal{H}[\pi_{\theta}(\cdot|s_t)] = \sum_j \mathcal{H}[\mathcal{N}(\mu_{t,j}, \sigma_{t,j})]
            probs.entropy().sum(1),
            self.critic(x),
        )
```

$$
\sigma_{\theta}(s_t) = \exp(\log \sigma_{\theta}(s_t))
$$

### Line 271

```python
            probs.log_prob(action).sum(1),
            # LaTeX: \mathcal{H}[\pi_{\theta}(\cdot|s_t)] = \sum_j \mathcal{H}[\mathcal{N}(\mu_{t,j}, \sigma_{t,j})]
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
```

$$
\log \pi_{\theta}(a_t|s_t) = \sum_j \log \mathcal{N}(a_{t,j}; \mu_{t,j}, \sigma_{t,j})
$$

### Line 273

```python
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
```

$$
\mathcal{H}[\pi_{\theta}(\cdot|s_t)] = \sum_j \mathcal{H}[\mathcal{N}(\mu_{t,j}, \sigma_{t,j})]
$$

### Line 413

```python
        eval_interval = max(1, total_steps // 150)

        # LaTeX: U = \left\lfloor \frac{T_{total}}{N \cdot T} \right\rfloor

        num_updates = total_steps // self.batch_size
        if num_updates <= 0:
            print(
                "[PPO] no updates scheduled because requested_total_steps < batch_size "
                f"({total_steps} < {self.batch_size})."
            )
            self.envs.close()
            self.eval_env.close()
```

$$
\Delta t_{eval} = \max\left(1, \left\lfloor \frac{T_{total}}{150} \right\rfloor\right)
$$

### Line 417

```python
        num_updates = total_steps // self.batch_size
        if num_updates <= 0:
            print(
                "[PPO] no updates scheduled because requested_total_steps < batch_size "
                f"({total_steps} < {self.batch_size})."
            )
            self.envs.close()
            self.eval_env.close()
            return

        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape,
```

$$
U = \left\lfloor \frac{T_{total}}{N \cdot T} \right\rfloor
$$

### Line 454

```python
                frac = 1.0 - (update - 1.0) / num_updates
                # LaTeX: \lambda_u = f_u \lambda_0
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                # LaTeX: t \leftarrow t + N
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
```

$$
f_u = 1 - \frac{u-1}{U}
$$

### Line 456

```python
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                # LaTeX: t \leftarrow t + N
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
```

$$
\lambda_u = f_u \lambda_0
$$

### Line 461

```python
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
```

$$
t \leftarrow t + N
$$

### Line 477

```python
                done = np.logical_or(terminated, truncated)

                rewards[step] = torch.as_tensor(
                    reward, dtype=torch.float32, device=self.device
                ).view(-1)
                next_obs = torch.as_tensor(
                    next_obs_np, dtype=torch.float32, device=self.device
                )
                next_done = torch.as_tensor(
                    done, dtype=torch.float32, device=self.device
                )
```

$$
d_t = d_t^{term} \lor d_t^{trunc}
$$

### Line 545

```python
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        # LaTeX: \delta_t = r_t + \gamma V(s_{t+1}) m_{t+1} - V(s_t)
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
```

$$
m_{t+1} = 1 - d_{t+1}
$$

### Line 549

```python
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        # LaTeX: \delta_t = r_t + \gamma V(s_{t+1}) m_{t+1} - V(s_t)
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        # LaTeX: A_t = \delta_t + \gamma \lambda m_{t+1} A_{t+1}
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
```

$$
m_{t+1} = 1 - d_{t+1}
$$

### Line 552

```python
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        # LaTeX: A_t = \delta_t + \gamma \lambda m_{t+1} A_{t+1}
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
```

$$
\delta_t = r_t + \gamma V(s_{t+1}) m_{t+1} - V(s_t)
$$

### Line 558

```python
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    # LaTeX: R_t = A_t + V(s_t)
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards, device=self.device)
                    for t in reversed(range(self.num_steps)):
```

$$
A_t = \delta_t + \gamma \lambda m_{t+1} A_{t+1}
$$

### Line 566

```python
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards, device=self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
```

$$
R_t = A_t + V(s_t)
$$

### Line 572

```python
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        # LaTeX: R_t = r_t + \gamma m_{t+1} R_{t+1}
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    # LaTeX: A_t = R_t - V(s_t)
                    advantages = returns - values
```

$$
m_{t+1} = 1 - d_{t+1}
$$

### Line 576

```python
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        # LaTeX: R_t = r_t + \gamma m_{t+1} R_{t+1}
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    # LaTeX: A_t = R_t - V(s_t)
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
```

$$
m_{t+1} = 1 - d_{t+1}
$$

### Line 579

```python
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    # LaTeX: A_t = R_t - V(s_t)
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
```

$$
R_t = r_t + \gamma m_{t+1} R_{t+1}
$$

### Line 583

```python
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []
```

$$
A_t = R_t - V(s_t)
$$

### Line 612

```python
                    logratio = newlogprob - b_logprobs[mb_inds]
                    # LaTeX: r_t = \exp(\log r_t)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # LaTeX: \widehat{D}_{KL}^{old} \approx \mathbb{E}[-\log r_t]
                        old_approx_kl = (-logratio).mean()
                        # LaTeX: \widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
```

$$
\log r_t = \log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t)
$$

### Line 614

```python
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # LaTeX: \widehat{D}_{KL}^{old} \approx \mathbb{E}[-\log r_t]
                        old_approx_kl = (-logratio).mean()
                        # LaTeX: \widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]
```

$$
r_t = \exp(\log r_t)
$$

### Line 619

```python
                        old_approx_kl = (-logratio).mean()
                        # LaTeX: \widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        # LaTeX: \hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
```

$$
\widehat{D}_{KL}^{old} \approx \mathbb{E}[-\log r_t]
$$

### Line 621

```python
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        # LaTeX: \hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
```

$$
\widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
$$

### Line 629

```python
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # LaTeX: \mathcal{L}_{pg}^{(1)} = -\hat{A}_t r_t

                    pg_loss1 = -mb_advantages * ratio
                    # LaTeX: \mathcal{L}_{pg}^{(2)} = -\hat{A}_t \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    # LaTeX: \mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
```

$$
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
$$

### Line 635

```python
                    pg_loss1 = -mb_advantages * ratio
                    # LaTeX: \mathcal{L}_{pg}^{(2)} = -\hat{A}_t \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    # LaTeX: \mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        # LaTeX: \mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
```

$$
\mathcal{L}_{pg}^{(1)} = -\hat{A}_t r_t
$$

### Line 637

```python
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    # LaTeX: \mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        # LaTeX: \mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # LaTeX: V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
                        v_clipped = b_values[mb_inds] + torch.clamp(
```

$$
\mathcal{L}_{pg}^{(2)} = -\hat{A}_t \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)
$$

### Line 641

```python
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        # LaTeX: \mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # LaTeX: V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
```

$$
\mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
$$

### Line 646

```python
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # LaTeX: V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        # LaTeX: \mathcal{L}_{V}^{clip} = (V_{\theta}^{clip}(s_t)-R_t)^2
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
                        v_loss = 0.5 * v_loss_max.mean()
```

$$
\mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
$$

### Line 648

```python
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        # LaTeX: \mathcal{L}_{V}^{clip} = (V_{\theta}^{clip}(s_t)-R_t)^2
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
```

$$
V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
$$

### Line 654

```python
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # LaTeX: \mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
```

$$
\mathcal{L}_{V}^{clip} = (V_{\theta}^{clip}(s_t)-R_t)^2
$$

### Line 657

```python
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # LaTeX: \mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
```

$$
\mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
$$

### Line 660

```python
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # LaTeX: \mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
```

$$
\mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
$$

### Line 664

```python
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None:
```

$$
\mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
$$

### Line 682

```python
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            sps = int(global_step / max(time.time() - start_time, 1e-8))
            log_wandb(
                {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
```

$$
\operatorname{EV} = 1 - \frac{\operatorname{Var}[R - V]}{\operatorname{Var}[R]}
$$

## Full Source

```python
from __future__ import annotations

import os
import random
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from minerva.utils.wandb_utils import log_wandb


def _transform_observation(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        return gym.wrappers.TransformObservation(env, fn)
    except TypeError:
        return gym.wrappers.TransformObservation(env, fn, env.observation_space)


def _transform_reward(env: gym.Env, fn):
    """Gymnasium compatibility shim across wrapper signatures."""
    try:
        return gym.wrappers.TransformReward(env, fn)
    except TypeError:
        return gym.wrappers.TransformReward(env, fn, env.reward_range)


def _resolve_env_id(env_id: str) -> str:
    if env_id.startswith("dm_control/"):
        parts = env_id.split("/")
        if len(parts) != 3:
            raise ValueError(
                "Expected dm_control env id format 'dm_control/<domain>/<task>', "
                f"got '{env_id}'"
            )
        _, domain, task = parts
        return f"dm_control/{domain}-{task}-v0"
    return env_id


def make_env(
    gym_id: str,
    seed: int,
    normalize_observation: bool = True,
):
    def thunk():
        resolved_env_id = _resolve_env_id(gym_id)
        env = gym.make(resolved_env_id)

        # Keep dm_control compatibility while preserving implementation-details PPO logic.
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
            env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = _transform_reward(env, lambda reward: np.clip(reward, -10, 10))

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_eval_env(gym_id: str, seed: int, normalize_observation: bool = True):
    def thunk():
        resolved_env_id = _resolve_env_id(gym_id)
        env = gym.make(resolved_env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize_observation:
            env = gym.wrappers.NormalizeObservation(env)
            env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    return None


def _sync_obs_rms_to_eval_envs(
    train_envs: gym.vector.VectorEnv, eval_envs: gym.vector.VectorEnv
):
    """Copy obs RMS stats from the first training env to all eval envs."""
    train_norm = find_wrapper(train_envs.envs[0], gym.wrappers.NormalizeObservation)
    if train_norm is None:
        return
    for eval_env in eval_envs.envs:
        eval_norm = find_wrapper(eval_env, gym.wrappers.NormalizeObservation)
        if eval_norm is not None:
            eval_norm.obs_rms.mean = np.copy(train_norm.obs_rms.mean)
            eval_norm.obs_rms.var = np.copy(train_norm.obs_rms.var)
            eval_norm.obs_rms.count = train_norm.obs_rms.count


@torch.no_grad()
def _evaluate_vectorized(
    agent: "Agent",
    eval_envs: gym.vector.VectorEnv,
    device: torch.device,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation: runs all episodes in parallel across eval_envs."""
    n_episodes = eval_envs.num_envs
    was_training = agent.training
    agent.eval()

    # Freeze obs normalization updates during eval.
    for env in eval_envs.envs:
        norm = find_wrapper(env, gym.wrappers.NormalizeObservation)
        if norm is not None and hasattr(norm, "update_running_mean"):
            norm.update_running_mean = False

    obs, _ = eval_envs.reset(seed=seed)
    episode_returns = np.zeros(n_episodes, dtype=np.float64)
    episode_lengths = np.zeros(n_episodes, dtype=np.int64)
    final_returns = []
    final_lengths = []
    done_mask = np.zeros(n_episodes, dtype=bool)

    while len(final_returns) < n_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action = agent.actor_mean(obs_t).cpu().numpy()
        obs, reward, terminated, truncated, _ = eval_envs.step(action)
        episode_returns += np.asarray(reward, dtype=np.float64)
        episode_lengths += 1
        done = np.asarray(terminated) | np.asarray(truncated)
        for i in range(n_episodes):
            if not done_mask[i] and done[i]:
                final_returns.append(float(episode_returns[i]))
                final_lengths.append(int(episode_lengths[i]))
                done_mask[i] = True

    # Re-enable obs normalization updates.
    for env in eval_envs.envs:
        norm = find_wrapper(env, gym.wrappers.NormalizeObservation)
        if norm is not None and hasattr(norm, "update_running_mean"):
            norm.update_running_mean = True

    if was_training:
        agent.train()

    return np.array(final_returns), np.array(final_lengths)


def log_episode_stats(infos, global_step: int):
    if not isinstance(infos, dict):
        return

    # Vector envs commonly expose episode stats as infos["episode"] with infos["_episode"] mask.
    if "episode" in infos:
        episode = infos["episode"]
        ep_returns = np.asarray(episode["r"]).reshape(-1)
        ep_lengths = np.asarray(episode["l"]).reshape(-1)
        ep_mask = np.asarray(
            infos.get("_episode", np.ones_like(ep_returns, dtype=bool))
        ).reshape(-1)
        for idx in np.where(ep_mask)[0]:
            episode_return = float(ep_returns[idx])
            episode_length = float(ep_lengths[idx])
            print(f"global_step={global_step}, episode_return={episode_return}")
            log_wandb(
                {
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_length,
                },
                step=global_step,
                silent=True,
            )

    # Some wrappers/setups expose terminal episode stats via final_info.
    elif "final_info" in infos:
        for item in infos["final_info"]:
            if item and "episode" in item:
                episode_return = float(np.asarray(item["episode"]["r"]).reshape(-1)[0])
                episode_length = float(np.asarray(item["episode"]["l"]).reshape(-1)[0])
                print(f"global_step={global_step}, episode_return={episode_return}")
                log_wandb(
                    {
                        "train/episode_return": episode_return,
                        "train/episode_length": episode_length,
                    },
                    step=global_step,
                    silent=True,
                )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_layer_sizes: Tuple[int, ...],
        value_layer_sizes: Tuple[int, ...],
    ):
        super().__init__()
        if len(policy_layer_sizes) == 0:
            raise ValueError("policy_layer_sizes must contain at least one layer size")
        if len(value_layer_sizes) == 0:
            raise ValueError("critic_layer_sizes must contain at least one layer size")

        self.critic = self._build_mlp(
            input_dim=obs_dim,
            hidden_layer_sizes=value_layer_sizes,
            output_dim=1,
            output_std=1.0,
        )
        self.actor_mean = self._build_mlp(
            input_dim=obs_dim,
            hidden_layer_sizes=policy_layer_sizes,
            output_dim=act_dim,
            output_std=0.01,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_layer_sizes: Tuple[int, ...],
        output_dim: int,
        output_std: float,
    ) -> nn.Sequential:
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.extend([layer_init(nn.Linear(last_dim, hidden_dim)), nn.Tanh()])
            last_dim = hidden_dim
        layers.append(layer_init(nn.Linear(last_dim, output_dim), std=output_std))
        return nn.Sequential(*layers)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # LaTeX: \sigma_{\theta}(s_t) = \exp(\log \sigma_{\theta}(s_t))
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            # LaTeX: \log \pi_{\theta}(a_t|s_t) = \sum_j \log \mathcal{N}(a_{t,j}; \mu_{t,j}, \sigma_{t,j})
            probs.log_prob(action).sum(1),
            # LaTeX: \mathcal{H}[\pi_{\theta}(\cdot|s_t)] = \sum_j \mathcal{H}[\mathcal{N}(\mu_{t,j}, \sigma_{t,j})]
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        rollout_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        policy_lr: float = 3e-4,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        anneal_lr: bool = True,
        normalize_obs: bool = True,
        num_envs: int = 1,
        optimizer_type: str = "adam",
        sgd_momentum: float = 0.9,
    ):
        self.env_id = str(env_id)
        self.seed = int(seed)
        self.device = device

        self.num_envs = int(num_envs)
        self.num_steps = int(rollout_steps)
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(minibatch_size)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.gae = True

        self.update_epochs = int(update_epochs)
        self.clip_coef = float(clip_ratio)
        self.clip_vloss = bool(clip_vloss)
        self.norm_adv = bool(norm_adv)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = (
            None if target_kl is None or float(target_kl) <= 0.0 else float(target_kl)
        )
        self.anneal_lr = bool(anneal_lr)

        self.learning_rate = float(policy_lr)
        self.optimizer_type = str(optimizer_type).strip().lower()
        self.sgd_momentum = float(sgd_momentum)

        self.normalize_obs = bool(normalize_obs)

        self.eval_episodes = 50
        self.eval_deterministic = True

        # Keep ppo-implementation-details behavior deterministic.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    self.env_id,
                    self.seed + i,
                    normalize_observation=self.normalize_obs,
                )
                for i in range(self.num_envs)
            ]
        )
        self.eval_envs = gym.vector.SyncVectorEnv(
            [
                make_eval_env(
                    self.env_id,
                    self.seed + 10_000 + i,
                    normalize_observation=self.normalize_obs,
                )
                for i in range(self.eval_episodes)
            ]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        if obs_shape is None:
            raise ValueError("observation space has no shape")
        if act_shape is None:
            raise ValueError("action space has no shape")

        obs_dim = int(np.array(obs_shape).prod())
        act_dim = int(np.prod(act_shape))

        self.agent = Agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=tuple(policy_layer_sizes),
            value_layer_sizes=tuple(critic_layer_sizes),
        ).to(self.device)

        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_type == "adam":
            return optim.Adam(
                self.agent.parameters(),
                lr=self.learning_rate,
                eps=1e-5,
            )
        if self.optimizer_type == "sgd":
            return optim.SGD(
                self.agent.parameters(),
                lr=self.learning_rate,
                momentum=self.sgd_momentum,
            )
        raise ValueError(
            f"Unsupported PPO optimizer_type '{self.optimizer_type}'. "
            "Expected one of: adam, sgd."
        )

    def train(
        self,
        total_steps: int,
        out_dir: str,
    ):
        total_steps = int(total_steps)
        # LaTeX: \Delta t_{eval} = \max\left(1, \left\lfloor \frac{T_{total}}{150} \right\rfloor\right)
        eval_interval = max(1, total_steps // 150)

        # LaTeX: U = \left\lfloor \frac{T_{total}}{N \cdot T} \right\rfloor

        num_updates = total_steps // self.batch_size
        if num_updates <= 0:
            print(
                "[PPO] no updates scheduled because requested_total_steps < batch_size "
                f"({total_steps} < {self.batch_size})."
            )
            self.envs.close()
            self.eval_env.close()
            return

        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape,
            device=self.device,
        )
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape,
            device=self.device,
        )
        logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        values = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        global_step = 0
        start_time = time.time()

        next_obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, device=self.device)

        last_eval = 0
        best_eval_score = float("-inf")
        os.makedirs(out_dir, exist_ok=True)

        for update in range(1, num_updates + 1):
            if self.anneal_lr:
                # LaTeX: f_u = 1 - \frac{u-1}{U}
                frac = 1.0 - (update - 1.0) / num_updates
                # LaTeX: \lambda_u = f_u \lambda_0
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                # LaTeX: t \leftarrow t + N
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                # LaTeX: d_t = d_t^{term} \lor d_t^{trunc}
                done = np.logical_or(terminated, truncated)

                rewards[step] = torch.as_tensor(
                    reward, dtype=torch.float32, device=self.device
                ).view(-1)
                next_obs = torch.as_tensor(
                    next_obs_np, dtype=torch.float32, device=self.device
                )
                next_done = torch.as_tensor(
                    done, dtype=torch.float32, device=self.device
                )

                log_episode_stats(infos, global_step)

                if global_step // eval_interval > last_eval:
                    last_eval = global_step // eval_interval
                    _sync_obs_rms_to_eval_envs(self.envs, self.eval_envs)
                    eval_returns, eval_lengths = _evaluate_vectorized(
                        self.agent,
                        self.eval_envs,
                        self.device,
                        seed=self.seed + 10_000,
                    )
                    metrics = {
                        "eval/return_max": float(np.max(eval_returns)),
                        "eval/return_std": float(np.std(eval_returns)),
                        "eval/return_mean": float(np.mean(eval_returns)),
                        "eval/length_mean": float(np.mean(eval_lengths)),
                        "eval/return_min": float(np.min(eval_returns)),
                    }
                    print(f"eval global_step={global_step}, " f"{metrics}")
                    log_wandb(
                        metrics,
                        step=global_step,
                        silent=True,
                    )

                    ckpt_payload = {
                        "actor_mean": self.agent.actor_mean.state_dict(),
                        "actor_logstd": self.agent.actor_logstd.detach().cpu(),
                        "critic": self.agent.critic.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    ckpt_last_path = os.path.join(out_dir, "ppo_last.pt")
                    torch.save(ckpt_payload, ckpt_last_path)
                    print(
                        f"[PPO][checkpoint] step={global_step}/{total_steps}: "
                        f"saved {ckpt_last_path}"
                    )

                    eval_score = float(metrics["eval/return_mean"])
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        ckpt_best_path = os.path.join(out_dir, "ppo_best.pt")
                        torch.save(ckpt_payload, ckpt_best_path)
                        print(
                            f"[PPO][checkpoint-best] step={global_step}/{total_steps}: "
                            f"score={eval_score:.6f}, saved {ckpt_best_path}"
                        )

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.gae:
                    advantages = torch.zeros_like(rewards, device=self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        # LaTeX: \delta_t = r_t + \gamma V(s_{t+1}) m_{t+1} - V(s_t)
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        # LaTeX: A_t = \delta_t + \gamma \lambda m_{t+1} A_{t+1}
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    # LaTeX: R_t = A_t + V(s_t)
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards, device=self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            # LaTeX: m_{t+1} = 1 - d_{t+1}
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        # LaTeX: R_t = r_t + \gamma m_{t+1} R_{t+1}
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    # LaTeX: A_t = R_t - V(s_t)
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []

            # Track last minibatch values for logging parity with reference implementation.
            pg_loss = torch.tensor(0.0, device=self.device)
            v_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            old_approx_kl = torch.tensor(0.0, device=self.device)
            approx_kl = torch.tensor(0.0, device=self.device)

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    # LaTeX: \log r_t = \log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    # LaTeX: r_t = \exp(\log r_t)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # LaTeX: \widehat{D}_{KL}^{old} \approx \mathbb{E}[-\log r_t]
                        old_approx_kl = (-logratio).mean()
                        # LaTeX: \widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        # LaTeX: \hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # LaTeX: \mathcal{L}_{pg}^{(1)} = -\hat{A}_t r_t

                    pg_loss1 = -mb_advantages * ratio
                    # LaTeX: \mathcal{L}_{pg}^{(2)} = -\hat{A}_t \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    # LaTeX: \mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        # LaTeX: \mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # LaTeX: V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        # LaTeX: \mathcal{L}_{V}^{clip} = (V_{\theta}^{clip}(s_t)-R_t)^2
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # LaTeX: \mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # LaTeX: \mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            # LaTeX: \operatorname{EV} = 1 - \frac{\operatorname{Var}[R - V]}{\operatorname{Var}[R]}
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            sps = int(global_step / max(time.time() - start_time, 1e-8))
            log_wandb(
                {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "losses/explained_variance": float(explained_var),
                    "charts/SPS": float(sps),
                },
                step=global_step,
                silent=True,
            )
            print("SPS:", sps)

        self.envs.close()
        self.eval_envs.close()
```
