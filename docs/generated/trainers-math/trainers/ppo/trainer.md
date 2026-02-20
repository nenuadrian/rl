# `trainers.ppo.trainer` Math-Annotated Source

_Source: `minerva/trainers/ppo/trainer.py`_

The full file is rendered in one continuous code-style block, with each `# LaTeX:` marker replaced inline by a rendered formula.

## Annotated Source

<div class="math-annotated-codeblock">
  <div class="math-annotated-code-line">from __future__ import annotations</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import os</div>
  <div class="math-annotated-code-line">import random</div>
  <div class="math-annotated-code-line">import time</div>
  <div class="math-annotated-code-line">from typing import Tuple</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import gymnasium as gym</div>
  <div class="math-annotated-code-line">import numpy as np</div>
  <div class="math-annotated-code-line">import torch</div>
  <div class="math-annotated-code-line">import torch.nn as nn</div>
  <div class="math-annotated-code-line">import torch.optim as optim</div>
  <div class="math-annotated-code-line">from torch.distributions.normal import Normal</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">from minerva.utils.wandb_utils import log_wandb</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def _transform_observation(env: gym.Env, fn):</div>
  <div class="math-annotated-code-line">    &quot;&quot;&quot;Gymnasium compatibility shim across wrapper signatures.&quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">    try:</div>
  <div class="math-annotated-code-line">        return gym.wrappers.TransformObservation(env, fn)</div>
  <div class="math-annotated-code-line">    except TypeError:</div>
  <div class="math-annotated-code-line">        return gym.wrappers.TransformObservation(env, fn, env.observation_space)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def _transform_reward(env: gym.Env, fn):</div>
  <div class="math-annotated-code-line">    &quot;&quot;&quot;Gymnasium compatibility shim across wrapper signatures.&quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">    try:</div>
  <div class="math-annotated-code-line">        return gym.wrappers.TransformReward(env, fn)</div>
  <div class="math-annotated-code-line">    except TypeError:</div>
  <div class="math-annotated-code-line">        return gym.wrappers.TransformReward(env, fn, env.reward_range)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def _resolve_env_id(env_id: str) -&gt; str:</div>
  <div class="math-annotated-code-line">    if env_id.startswith(&quot;dm_control/&quot;):</div>
  <div class="math-annotated-code-line">        parts = env_id.split(&quot;/&quot;)</div>
  <div class="math-annotated-code-line">        if len(parts) != 3:</div>
  <div class="math-annotated-code-line">            raise ValueError(</div>
  <div class="math-annotated-code-line">                &quot;Expected dm_control env id format &#x27;dm_control/&lt;domain&gt;/&lt;task&gt;&#x27;, &quot;</div>
  <div class="math-annotated-code-line">                f&quot;got &#x27;{env_id}&#x27;&quot;</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">        _, domain, task = parts</div>
  <div class="math-annotated-code-line">        return f&quot;dm_control/{domain}-{task}-v0&quot;</div>
  <div class="math-annotated-code-line">    return env_id</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def make_env(</div>
  <div class="math-annotated-code-line">    gym_id: str,</div>
  <div class="math-annotated-code-line">    seed: int,</div>
  <div class="math-annotated-code-line">    normalize_observation: bool = True,</div>
  <div class="math-annotated-code-line">):</div>
  <div class="math-annotated-code-line">    def thunk():</div>
  <div class="math-annotated-code-line">        resolved_env_id = _resolve_env_id(gym_id)</div>
  <div class="math-annotated-code-line">        env = gym.make(resolved_env_id)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Keep dm_control compatibility while preserving implementation-details PPO logic.</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.FlattenObservation(env)</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.RecordEpisodeStatistics(env)</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.ClipAction(env)</div>
  <div class="math-annotated-code-line">        if normalize_observation:</div>
  <div class="math-annotated-code-line">            env = gym.wrappers.NormalizeObservation(env)</div>
  <div class="math-annotated-code-line">            env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.NormalizeReward(env)</div>
  <div class="math-annotated-code-line">        env = _transform_reward(env, lambda reward: np.clip(reward, -10, 10))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        env.reset(seed=seed)</div>
  <div class="math-annotated-code-line">        env.action_space.seed(seed)</div>
  <div class="math-annotated-code-line">        env.observation_space.seed(seed)</div>
  <div class="math-annotated-code-line">        return env</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    return thunk</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def make_eval_env(gym_id: str, seed: int, normalize_observation: bool = True):</div>
  <div class="math-annotated-code-line">    def thunk():</div>
  <div class="math-annotated-code-line">        resolved_env_id = _resolve_env_id(gym_id)</div>
  <div class="math-annotated-code-line">        env = gym.make(resolved_env_id)</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.FlattenObservation(env)</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.RecordEpisodeStatistics(env)</div>
  <div class="math-annotated-code-line">        env = gym.wrappers.ClipAction(env)</div>
  <div class="math-annotated-code-line">        if normalize_observation:</div>
  <div class="math-annotated-code-line">            env = gym.wrappers.NormalizeObservation(env)</div>
  <div class="math-annotated-code-line">            env = _transform_observation(env, lambda obs: np.clip(obs, -10, 10))</div>
  <div class="math-annotated-code-line">        env.reset(seed=seed)</div>
  <div class="math-annotated-code-line">        env.action_space.seed(seed)</div>
  <div class="math-annotated-code-line">        env.observation_space.seed(seed)</div>
  <div class="math-annotated-code-line">        return env</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    return thunk</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def find_wrapper(env, wrapper_type):</div>
  <div class="math-annotated-code-line">    current = env</div>
  <div class="math-annotated-code-line">    while current is not None:</div>
  <div class="math-annotated-code-line">        if isinstance(current, wrapper_type):</div>
  <div class="math-annotated-code-line">            return current</div>
  <div class="math-annotated-code-line">        current = getattr(current, &quot;env&quot;, None)</div>
  <div class="math-annotated-code-line">    return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def _sync_obs_rms_to_eval_envs(</div>
  <div class="math-annotated-code-line">    train_envs: gym.vector.VectorEnv, eval_envs: gym.vector.VectorEnv</div>
  <div class="math-annotated-code-line">):</div>
  <div class="math-annotated-code-line">    &quot;&quot;&quot;Copy obs RMS stats from the first training env to all eval envs.&quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">    train_norm = find_wrapper(train_envs.envs[0], gym.wrappers.NormalizeObservation)</div>
  <div class="math-annotated-code-line">    if train_norm is None:</div>
  <div class="math-annotated-code-line">        return</div>
  <div class="math-annotated-code-line">    for eval_env in eval_envs.envs:</div>
  <div class="math-annotated-code-line">        eval_norm = find_wrapper(eval_env, gym.wrappers.NormalizeObservation)</div>
  <div class="math-annotated-code-line">        if eval_norm is not None:</div>
  <div class="math-annotated-code-line">            eval_norm.obs_rms.mean = np.copy(train_norm.obs_rms.mean)</div>
  <div class="math-annotated-code-line">            eval_norm.obs_rms.var = np.copy(train_norm.obs_rms.var)</div>
  <div class="math-annotated-code-line">            eval_norm.obs_rms.count = train_norm.obs_rms.count</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">@torch.no_grad()</div>
  <div class="math-annotated-code-line">def _evaluate_vectorized(</div>
  <div class="math-annotated-code-line">    agent: &quot;Agent&quot;,</div>
  <div class="math-annotated-code-line">    eval_envs: gym.vector.VectorEnv,</div>
  <div class="math-annotated-code-line">    device: torch.device,</div>
  <div class="math-annotated-code-line">    seed: int = 42,</div>
  <div class="math-annotated-code-line">) -&gt; tuple[np.ndarray, np.ndarray]:</div>
  <div class="math-annotated-code-line">    &quot;&quot;&quot;Vectorized evaluation: runs all episodes in parallel across eval_envs.&quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">    n_episodes = eval_envs.num_envs</div>
  <div class="math-annotated-code-line">    was_training = agent.training</div>
  <div class="math-annotated-code-line">    agent.eval()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    # Freeze obs normalization updates during eval.</div>
  <div class="math-annotated-code-line">    for env in eval_envs.envs:</div>
  <div class="math-annotated-code-line">        norm = find_wrapper(env, gym.wrappers.NormalizeObservation)</div>
  <div class="math-annotated-code-line">        if norm is not None and hasattr(norm, &quot;update_running_mean&quot;):</div>
  <div class="math-annotated-code-line">            norm.update_running_mean = False</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    obs, _ = eval_envs.reset(seed=seed)</div>
  <div class="math-annotated-code-line">    episode_returns = np.zeros(n_episodes, dtype=np.float64)</div>
  <div class="math-annotated-code-line">    episode_lengths = np.zeros(n_episodes, dtype=np.int64)</div>
  <div class="math-annotated-code-line">    final_returns = []</div>
  <div class="math-annotated-code-line">    final_lengths = []</div>
  <div class="math-annotated-code-line">    done_mask = np.zeros(n_episodes, dtype=bool)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    while len(final_returns) &lt; n_episodes:</div>
  <div class="math-annotated-code-line">        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)</div>
  <div class="math-annotated-code-line">        action = agent.actor_mean(obs_t).cpu().numpy()</div>
  <div class="math-annotated-code-line">        obs, reward, terminated, truncated, _ = eval_envs.step(action)</div>
  <div class="math-annotated-code-line">        episode_returns += np.asarray(reward, dtype=np.float64)</div>
  <div class="math-annotated-code-line">        episode_lengths += 1</div>
  <div class="math-annotated-code-line">        done = np.asarray(terminated) | np.asarray(truncated)</div>
  <div class="math-annotated-code-line">        for i in range(n_episodes):</div>
  <div class="math-annotated-code-line">            if not done_mask[i] and done[i]:</div>
  <div class="math-annotated-code-line">                final_returns.append(float(episode_returns[i]))</div>
  <div class="math-annotated-code-line">                final_lengths.append(int(episode_lengths[i]))</div>
  <div class="math-annotated-code-line">                done_mask[i] = True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    # Re-enable obs normalization updates.</div>
  <div class="math-annotated-code-line">    for env in eval_envs.envs:</div>
  <div class="math-annotated-code-line">        norm = find_wrapper(env, gym.wrappers.NormalizeObservation)</div>
  <div class="math-annotated-code-line">        if norm is not None and hasattr(norm, &quot;update_running_mean&quot;):</div>
  <div class="math-annotated-code-line">            norm.update_running_mean = True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    if was_training:</div>
  <div class="math-annotated-code-line">        agent.train()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    return np.array(final_returns), np.array(final_lengths)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def log_episode_stats(infos, global_step: int):</div>
  <div class="math-annotated-code-line">    if not isinstance(infos, dict):</div>
  <div class="math-annotated-code-line">        return</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    # Vector envs commonly expose episode stats as infos[&quot;episode&quot;] with infos[&quot;_episode&quot;] mask.</div>
  <div class="math-annotated-code-line">    if &quot;episode&quot; in infos:</div>
  <div class="math-annotated-code-line">        episode = infos[&quot;episode&quot;]</div>
  <div class="math-annotated-code-line">        ep_returns = np.asarray(episode[&quot;r&quot;]).reshape(-1)</div>
  <div class="math-annotated-code-line">        ep_lengths = np.asarray(episode[&quot;l&quot;]).reshape(-1)</div>
  <div class="math-annotated-code-line">        ep_mask = np.asarray(</div>
  <div class="math-annotated-code-line">            infos.get(&quot;_episode&quot;, np.ones_like(ep_returns, dtype=bool))</div>
  <div class="math-annotated-code-line">        ).reshape(-1)</div>
  <div class="math-annotated-code-line">        for idx in np.where(ep_mask)[0]:</div>
  <div class="math-annotated-code-line">            episode_return = float(ep_returns[idx])</div>
  <div class="math-annotated-code-line">            episode_length = float(ep_lengths[idx])</div>
  <div class="math-annotated-code-line">            print(f&quot;global_step={global_step}, episode_return={episode_return}&quot;)</div>
  <div class="math-annotated-code-line">            log_wandb(</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;train/episode_return&quot;: episode_return,</div>
  <div class="math-annotated-code-line">                    &quot;train/episode_length&quot;: episode_length,</div>
  <div class="math-annotated-code-line">                },</div>
  <div class="math-annotated-code-line">                step=global_step,</div>
  <div class="math-annotated-code-line">                silent=True,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    # Some wrappers/setups expose terminal episode stats via final_info.</div>
  <div class="math-annotated-code-line">    elif &quot;final_info&quot; in infos:</div>
  <div class="math-annotated-code-line">        for item in infos[&quot;final_info&quot;]:</div>
  <div class="math-annotated-code-line">            if item and &quot;episode&quot; in item:</div>
  <div class="math-annotated-code-line">                episode_return = float(np.asarray(item[&quot;episode&quot;][&quot;r&quot;]).reshape(-1)[0])</div>
  <div class="math-annotated-code-line">                episode_length = float(np.asarray(item[&quot;episode&quot;][&quot;l&quot;]).reshape(-1)[0])</div>
  <div class="math-annotated-code-line">                print(f&quot;global_step={global_step}, episode_return={episode_return}&quot;)</div>
  <div class="math-annotated-code-line">                log_wandb(</div>
  <div class="math-annotated-code-line">                    {</div>
  <div class="math-annotated-code-line">                        &quot;train/episode_return&quot;: episode_return,</div>
  <div class="math-annotated-code-line">                        &quot;train/episode_length&quot;: episode_length,</div>
  <div class="math-annotated-code-line">                    },</div>
  <div class="math-annotated-code-line">                    step=global_step,</div>
  <div class="math-annotated-code-line">                    silent=True,</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">def layer_init(layer, std=np.sqrt(2), bias_const=0.0):</div>
  <div class="math-annotated-code-line">    torch.nn.init.orthogonal_(layer.weight, std)</div>
  <div class="math-annotated-code-line">    torch.nn.init.constant_(layer.bias, bias_const)</div>
  <div class="math-annotated-code-line">    return layer</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class Agent(nn.Module):</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs_dim: int,</div>
  <div class="math-annotated-code-line">        act_dim: int,</div>
  <div class="math-annotated-code-line">        policy_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        value_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        super().__init__()</div>
  <div class="math-annotated-code-line">        if len(policy_layer_sizes) == 0:</div>
  <div class="math-annotated-code-line">            raise ValueError(&quot;policy_layer_sizes must contain at least one layer size&quot;)</div>
  <div class="math-annotated-code-line">        if len(value_layer_sizes) == 0:</div>
  <div class="math-annotated-code-line">            raise ValueError(&quot;critic_layer_sizes must contain at least one layer size&quot;)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.critic = self._build_mlp(</div>
  <div class="math-annotated-code-line">            input_dim=obs_dim,</div>
  <div class="math-annotated-code-line">            hidden_layer_sizes=value_layer_sizes,</div>
  <div class="math-annotated-code-line">            output_dim=1,</div>
  <div class="math-annotated-code-line">            output_std=1.0,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.actor_mean = self._build_mlp(</div>
  <div class="math-annotated-code-line">            input_dim=obs_dim,</div>
  <div class="math-annotated-code-line">            hidden_layer_sizes=policy_layer_sizes,</div>
  <div class="math-annotated-code-line">            output_dim=act_dim,</div>
  <div class="math-annotated-code-line">            output_std=0.01,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    @staticmethod</div>
  <div class="math-annotated-code-line">    def _build_mlp(</div>
  <div class="math-annotated-code-line">        input_dim: int,</div>
  <div class="math-annotated-code-line">        hidden_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        output_dim: int,</div>
  <div class="math-annotated-code-line">        output_std: float,</div>
  <div class="math-annotated-code-line">    ) -&gt; nn.Sequential:</div>
  <div class="math-annotated-code-line">        layers = []</div>
  <div class="math-annotated-code-line">        last_dim = input_dim</div>
  <div class="math-annotated-code-line">        for hidden_dim in hidden_layer_sizes:</div>
  <div class="math-annotated-code-line">            layers.extend([layer_init(nn.Linear(last_dim, hidden_dim)), nn.Tanh()])</div>
  <div class="math-annotated-code-line">            last_dim = hidden_dim</div>
  <div class="math-annotated-code-line">        layers.append(layer_init(nn.Linear(last_dim, output_dim), std=output_std))</div>
  <div class="math-annotated-code-line">        return nn.Sequential(*layers)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def get_value(self, x):</div>
  <div class="math-annotated-code-line">        return self.critic(x)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def get_action_and_value(self, x, action=None):</div>
  <div class="math-annotated-code-line">        action_mean = self.actor_mean(x)</div>
  <div class="math-annotated-code-line">        action_logstd = self.actor_logstd.expand_as(action_mean)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\sigma_{\theta}(s_t) = \exp(\log \sigma_{\theta}(s_t))
\]</div>
  </div>
  <div class="math-annotated-code-line">        action_std = torch.exp(action_logstd)</div>
  <div class="math-annotated-code-line">        probs = Normal(action_mean, action_std)</div>
  <div class="math-annotated-code-line">        if action is None:</div>
  <div class="math-annotated-code-line">            action = probs.sample()</div>
  <div class="math-annotated-code-line">        return (</div>
  <div class="math-annotated-code-line">            action,</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\log \pi_{\theta}(a_t|s_t) = \sum_j \log \mathcal{N}(a_{t,j}; \mu_{t,j}, \sigma_{t,j})
\]</div>
  </div>
  <div class="math-annotated-code-line">            probs.log_prob(action).sum(1),</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{H}[\pi_{\theta}(\cdot|s_t)] = \sum_j \mathcal{H}[\mathcal{N}(\mu_{t,j}, \sigma_{t,j})]
\]</div>
  </div>
  <div class="math-annotated-code-line">            probs.entropy().sum(1),</div>
  <div class="math-annotated-code-line">            self.critic(x),</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class PPOTrainer:</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        env_id: str,</div>
  <div class="math-annotated-code-line">        seed: int,</div>
  <div class="math-annotated-code-line">        device: torch.device,</div>
  <div class="math-annotated-code-line">        policy_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        critic_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        rollout_steps: int,</div>
  <div class="math-annotated-code-line">        gamma: float = 0.99,</div>
  <div class="math-annotated-code-line">        gae_lambda: float = 0.95,</div>
  <div class="math-annotated-code-line">        update_epochs: int = 10,</div>
  <div class="math-annotated-code-line">        minibatch_size: int = 64,</div>
  <div class="math-annotated-code-line">        policy_lr: float = 3e-4,</div>
  <div class="math-annotated-code-line">        clip_ratio: float = 0.2,</div>
  <div class="math-annotated-code-line">        ent_coef: float = 0.0,</div>
  <div class="math-annotated-code-line">        vf_coef: float = 0.5,</div>
  <div class="math-annotated-code-line">        max_grad_norm: float = 0.5,</div>
  <div class="math-annotated-code-line">        target_kl: float = 0.02,</div>
  <div class="math-annotated-code-line">        norm_adv: bool = True,</div>
  <div class="math-annotated-code-line">        clip_vloss: bool = True,</div>
  <div class="math-annotated-code-line">        anneal_lr: bool = True,</div>
  <div class="math-annotated-code-line">        normalize_obs: bool = True,</div>
  <div class="math-annotated-code-line">        num_envs: int = 1,</div>
  <div class="math-annotated-code-line">        optimizer_type: str = &quot;adam&quot;,</div>
  <div class="math-annotated-code-line">        sgd_momentum: float = 0.9,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        self.env_id = str(env_id)</div>
  <div class="math-annotated-code-line">        self.seed = int(seed)</div>
  <div class="math-annotated-code-line">        self.device = device</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.num_envs = int(num_envs)</div>
  <div class="math-annotated-code-line">        self.num_steps = int(rollout_steps)</div>
  <div class="math-annotated-code-line">        self.batch_size = int(self.num_envs * self.num_steps)</div>
  <div class="math-annotated-code-line">        self.minibatch_size = int(minibatch_size)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.gamma = float(gamma)</div>
  <div class="math-annotated-code-line">        self.gae_lambda = float(gae_lambda)</div>
  <div class="math-annotated-code-line">        self.gae = True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.update_epochs = int(update_epochs)</div>
  <div class="math-annotated-code-line">        self.clip_coef = float(clip_ratio)</div>
  <div class="math-annotated-code-line">        self.clip_vloss = bool(clip_vloss)</div>
  <div class="math-annotated-code-line">        self.norm_adv = bool(norm_adv)</div>
  <div class="math-annotated-code-line">        self.ent_coef = float(ent_coef)</div>
  <div class="math-annotated-code-line">        self.vf_coef = float(vf_coef)</div>
  <div class="math-annotated-code-line">        self.max_grad_norm = float(max_grad_norm)</div>
  <div class="math-annotated-code-line">        self.target_kl = (</div>
  <div class="math-annotated-code-line">            None if target_kl is None or float(target_kl) &lt;= 0.0 else float(target_kl)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.anneal_lr = bool(anneal_lr)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.learning_rate = float(policy_lr)</div>
  <div class="math-annotated-code-line">        self.optimizer_type = str(optimizer_type).strip().lower()</div>
  <div class="math-annotated-code-line">        self.sgd_momentum = float(sgd_momentum)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.normalize_obs = bool(normalize_obs)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.eval_episodes = 50</div>
  <div class="math-annotated-code-line">        self.eval_deterministic = True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Keep ppo-implementation-details behavior deterministic.</div>
  <div class="math-annotated-code-line">        random.seed(self.seed)</div>
  <div class="math-annotated-code-line">        np.random.seed(self.seed)</div>
  <div class="math-annotated-code-line">        torch.manual_seed(self.seed)</div>
  <div class="math-annotated-code-line">        torch.backends.cudnn.deterministic = True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.envs = gym.vector.SyncVectorEnv(</div>
  <div class="math-annotated-code-line">            [</div>
  <div class="math-annotated-code-line">                make_env(</div>
  <div class="math-annotated-code-line">                    self.env_id,</div>
  <div class="math-annotated-code-line">                    self.seed + i,</div>
  <div class="math-annotated-code-line">                    normalize_observation=self.normalize_obs,</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                for i in range(self.num_envs)</div>
  <div class="math-annotated-code-line">            ]</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.eval_envs = gym.vector.SyncVectorEnv(</div>
  <div class="math-annotated-code-line">            [</div>
  <div class="math-annotated-code-line">                make_eval_env(</div>
  <div class="math-annotated-code-line">                    self.env_id,</div>
  <div class="math-annotated-code-line">                    self.seed + 10_000 + i,</div>
  <div class="math-annotated-code-line">                    normalize_observation=self.normalize_obs,</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                for i in range(self.eval_episodes)</div>
  <div class="math-annotated-code-line">            ]</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        assert isinstance(</div>
  <div class="math-annotated-code-line">            self.envs.single_action_space, gym.spaces.Box</div>
  <div class="math-annotated-code-line">        ), &quot;only continuous action space is supported&quot;</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs_shape = self.envs.single_observation_space.shape</div>
  <div class="math-annotated-code-line">        act_shape = self.envs.single_action_space.shape</div>
  <div class="math-annotated-code-line">        if obs_shape is None:</div>
  <div class="math-annotated-code-line">            raise ValueError(&quot;observation space has no shape&quot;)</div>
  <div class="math-annotated-code-line">        if act_shape is None:</div>
  <div class="math-annotated-code-line">            raise ValueError(&quot;action space has no shape&quot;)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs_dim = int(np.array(obs_shape).prod())</div>
  <div class="math-annotated-code-line">        act_dim = int(np.prod(act_shape))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.agent = Agent(</div>
  <div class="math-annotated-code-line">            obs_dim=obs_dim,</div>
  <div class="math-annotated-code-line">            act_dim=act_dim,</div>
  <div class="math-annotated-code-line">            policy_layer_sizes=tuple(policy_layer_sizes),</div>
  <div class="math-annotated-code-line">            value_layer_sizes=tuple(critic_layer_sizes),</div>
  <div class="math-annotated-code-line">        ).to(self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.optimizer = self._build_optimizer()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _build_optimizer(self) -&gt; torch.optim.Optimizer:</div>
  <div class="math-annotated-code-line">        if self.optimizer_type == &quot;adam&quot;:</div>
  <div class="math-annotated-code-line">            return optim.Adam(</div>
  <div class="math-annotated-code-line">                self.agent.parameters(),</div>
  <div class="math-annotated-code-line">                lr=self.learning_rate,</div>
  <div class="math-annotated-code-line">                eps=1e-5,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">        if self.optimizer_type == &quot;sgd&quot;:</div>
  <div class="math-annotated-code-line">            return optim.SGD(</div>
  <div class="math-annotated-code-line">                self.agent.parameters(),</div>
  <div class="math-annotated-code-line">                lr=self.learning_rate,</div>
  <div class="math-annotated-code-line">                momentum=self.sgd_momentum,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">        raise ValueError(</div>
  <div class="math-annotated-code-line">            f&quot;Unsupported PPO optimizer_type &#x27;{self.optimizer_type}&#x27;. &quot;</div>
  <div class="math-annotated-code-line">            &quot;Expected one of: adam, sgd.&quot;</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def train(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        total_steps: int,</div>
  <div class="math-annotated-code-line">        out_dir: str,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        total_steps = int(total_steps)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\Delta t_{eval} = \max\left(1, \left\lfloor \frac{T_{total}}{150} \right\rfloor\right)
\]</div>
  </div>
  <div class="math-annotated-code-line">        eval_interval = max(1, total_steps // 150)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
U = \left\lfloor \frac{T_{total}}{N \cdot T} \right\rfloor
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        num_updates = total_steps // self.batch_size</div>
  <div class="math-annotated-code-line">        if num_updates &lt;= 0:</div>
  <div class="math-annotated-code-line">            print(</div>
  <div class="math-annotated-code-line">                &quot;[PPO] no updates scheduled because requested_total_steps &lt; batch_size &quot;</div>
  <div class="math-annotated-code-line">                f&quot;({total_steps} &lt; {self.batch_size}).&quot;</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            self.envs.close()</div>
  <div class="math-annotated-code-line">            self.eval_env.close()</div>
  <div class="math-annotated-code-line">            return</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs = torch.zeros(</div>
  <div class="math-annotated-code-line">            (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape,</div>
  <div class="math-annotated-code-line">            device=self.device,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        actions = torch.zeros(</div>
  <div class="math-annotated-code-line">            (self.num_steps, self.num_envs) + self.envs.single_action_space.shape,</div>
  <div class="math-annotated-code-line">            device=self.device,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device)</div>
  <div class="math-annotated-code-line">        rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)</div>
  <div class="math-annotated-code-line">        dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)</div>
  <div class="math-annotated-code-line">        values = torch.zeros((self.num_steps, self.num_envs), device=self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        global_step = 0</div>
  <div class="math-annotated-code-line">        start_time = time.time()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        next_obs, _ = self.envs.reset()</div>
  <div class="math-annotated-code-line">        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)</div>
  <div class="math-annotated-code-line">        next_done = torch.zeros(self.num_envs, device=self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        last_eval = 0</div>
  <div class="math-annotated-code-line">        best_eval_score = float(&quot;-inf&quot;)</div>
  <div class="math-annotated-code-line">        os.makedirs(out_dir, exist_ok=True)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        for update in range(1, num_updates + 1):</div>
  <div class="math-annotated-code-line">            if self.anneal_lr:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
f_u = 1 - \frac{u-1}{U}
\]</div>
  </div>
  <div class="math-annotated-code-line">                frac = 1.0 - (update - 1.0) / num_updates</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\lambda_u = f_u \lambda_0
\]</div>
  </div>
  <div class="math-annotated-code-line">                lrnow = frac * self.learning_rate</div>
  <div class="math-annotated-code-line">                self.optimizer.param_groups[0][&quot;lr&quot;] = lrnow</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            for step in range(0, self.num_steps):</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
t \leftarrow t + N
\]</div>
  </div>
  <div class="math-annotated-code-line">                global_step += self.num_envs</div>
  <div class="math-annotated-code-line">                obs[step] = next_obs</div>
  <div class="math-annotated-code-line">                dones[step] = next_done</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                with torch.no_grad():</div>
  <div class="math-annotated-code-line">                    action, logprob, _, value = self.agent.get_action_and_value(</div>
  <div class="math-annotated-code-line">                        next_obs</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                    values[step] = value.flatten()</div>
  <div class="math-annotated-code-line">                actions[step] = action</div>
  <div class="math-annotated-code-line">                logprobs[step] = logprob</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                next_obs_np, reward, terminated, truncated, infos = self.envs.step(</div>
  <div class="math-annotated-code-line">                    action.cpu().numpy()</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
d_t = d_t^{term} \lor d_t^{trunc}
\]</div>
  </div>
  <div class="math-annotated-code-line">                done = np.logical_or(terminated, truncated)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                rewards[step] = torch.as_tensor(</div>
  <div class="math-annotated-code-line">                    reward, dtype=torch.float32, device=self.device</div>
  <div class="math-annotated-code-line">                ).view(-1)</div>
  <div class="math-annotated-code-line">                next_obs = torch.as_tensor(</div>
  <div class="math-annotated-code-line">                    next_obs_np, dtype=torch.float32, device=self.device</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                next_done = torch.as_tensor(</div>
  <div class="math-annotated-code-line">                    done, dtype=torch.float32, device=self.device</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                log_episode_stats(infos, global_step)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                if global_step // eval_interval &gt; last_eval:</div>
  <div class="math-annotated-code-line">                    last_eval = global_step // eval_interval</div>
  <div class="math-annotated-code-line">                    _sync_obs_rms_to_eval_envs(self.envs, self.eval_envs)</div>
  <div class="math-annotated-code-line">                    eval_returns, eval_lengths = _evaluate_vectorized(</div>
  <div class="math-annotated-code-line">                        self.agent,</div>
  <div class="math-annotated-code-line">                        self.eval_envs,</div>
  <div class="math-annotated-code-line">                        self.device,</div>
  <div class="math-annotated-code-line">                        seed=self.seed + 10_000,</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                    metrics = {</div>
  <div class="math-annotated-code-line">                        &quot;eval/return_max&quot;: float(np.max(eval_returns)),</div>
  <div class="math-annotated-code-line">                        &quot;eval/return_std&quot;: float(np.std(eval_returns)),</div>
  <div class="math-annotated-code-line">                        &quot;eval/return_mean&quot;: float(np.mean(eval_returns)),</div>
  <div class="math-annotated-code-line">                        &quot;eval/length_mean&quot;: float(np.mean(eval_lengths)),</div>
  <div class="math-annotated-code-line">                        &quot;eval/return_min&quot;: float(np.min(eval_returns)),</div>
  <div class="math-annotated-code-line">                    }</div>
  <div class="math-annotated-code-line">                    print(f&quot;eval global_step={global_step}, &quot; f&quot;{metrics}&quot;)</div>
  <div class="math-annotated-code-line">                    log_wandb(</div>
  <div class="math-annotated-code-line">                        metrics,</div>
  <div class="math-annotated-code-line">                        step=global_step,</div>
  <div class="math-annotated-code-line">                        silent=True,</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    ckpt_payload = {</div>
  <div class="math-annotated-code-line">                        &quot;actor_mean&quot;: self.agent.actor_mean.state_dict(),</div>
  <div class="math-annotated-code-line">                        &quot;actor_logstd&quot;: self.agent.actor_logstd.detach().cpu(),</div>
  <div class="math-annotated-code-line">                        &quot;critic&quot;: self.agent.critic.state_dict(),</div>
  <div class="math-annotated-code-line">                        &quot;optimizer&quot;: self.optimizer.state_dict(),</div>
  <div class="math-annotated-code-line">                    }</div>
  <div class="math-annotated-code-line">                    ckpt_last_path = os.path.join(out_dir, &quot;ppo_last.pt&quot;)</div>
  <div class="math-annotated-code-line">                    torch.save(ckpt_payload, ckpt_last_path)</div>
  <div class="math-annotated-code-line">                    print(</div>
  <div class="math-annotated-code-line">                        f&quot;[PPO][checkpoint] step={global_step}/{total_steps}: &quot;</div>
  <div class="math-annotated-code-line">                        f&quot;saved {ckpt_last_path}&quot;</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    eval_score = float(metrics[&quot;eval/return_mean&quot;])</div>
  <div class="math-annotated-code-line">                    if eval_score &gt; best_eval_score:</div>
  <div class="math-annotated-code-line">                        best_eval_score = eval_score</div>
  <div class="math-annotated-code-line">                        ckpt_best_path = os.path.join(out_dir, &quot;ppo_best.pt&quot;)</div>
  <div class="math-annotated-code-line">                        torch.save(ckpt_payload, ckpt_best_path)</div>
  <div class="math-annotated-code-line">                        print(</div>
  <div class="math-annotated-code-line">                            f&quot;[PPO][checkpoint-best] step={global_step}/{total_steps}: &quot;</div>
  <div class="math-annotated-code-line">                            f&quot;score={eval_score:.6f}, saved {ckpt_best_path}&quot;</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            with torch.no_grad():</div>
  <div class="math-annotated-code-line">                next_value = self.agent.get_value(next_obs).reshape(1, -1)</div>
  <div class="math-annotated-code-line">                if self.gae:</div>
  <div class="math-annotated-code-line">                    advantages = torch.zeros_like(rewards, device=self.device)</div>
  <div class="math-annotated-code-line">                    lastgaelam = 0</div>
  <div class="math-annotated-code-line">                    for t in reversed(range(self.num_steps)):</div>
  <div class="math-annotated-code-line">                        if t == self.num_steps - 1:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_{t+1} = 1 - d_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                            nextnonterminal = 1.0 - next_done</div>
  <div class="math-annotated-code-line">                            nextvalues = next_value</div>
  <div class="math-annotated-code-line">                        else:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_{t+1} = 1 - d_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                            nextnonterminal = 1.0 - dones[t + 1]</div>
  <div class="math-annotated-code-line">                            nextvalues = values[t + 1]</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\delta_t = r_t + \gamma V(s_{t+1}) m_{t+1} - V(s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">                        delta = (</div>
  <div class="math-annotated-code-line">                            rewards[t]</div>
  <div class="math-annotated-code-line">                            + self.gamma * nextvalues * nextnonterminal</div>
  <div class="math-annotated-code-line">                            - values[t]</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
A_t = \delta_t + \gamma \lambda m_{t+1} A_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                        advantages[t] = lastgaelam = (</div>
  <div class="math-annotated-code-line">                            delta</div>
  <div class="math-annotated-code-line">                            + self.gamma</div>
  <div class="math-annotated-code-line">                            * self.gae_lambda</div>
  <div class="math-annotated-code-line">                            * nextnonterminal</div>
  <div class="math-annotated-code-line">                            * lastgaelam</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
R_t = A_t + V(s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">                    returns = advantages + values</div>
  <div class="math-annotated-code-line">                else:</div>
  <div class="math-annotated-code-line">                    returns = torch.zeros_like(rewards, device=self.device)</div>
  <div class="math-annotated-code-line">                    for t in reversed(range(self.num_steps)):</div>
  <div class="math-annotated-code-line">                        if t == self.num_steps - 1:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_{t+1} = 1 - d_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                            nextnonterminal = 1.0 - next_done</div>
  <div class="math-annotated-code-line">                            next_return = next_value</div>
  <div class="math-annotated-code-line">                        else:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_{t+1} = 1 - d_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                            nextnonterminal = 1.0 - dones[t + 1]</div>
  <div class="math-annotated-code-line">                            next_return = returns[t + 1]</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
R_t = r_t + \gamma m_{t+1} R_{t+1}
\]</div>
  </div>
  <div class="math-annotated-code-line">                        returns[t] = (</div>
  <div class="math-annotated-code-line">                            rewards[t] + self.gamma * nextnonterminal * next_return</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
A_t = R_t - V(s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">                    advantages = returns - values</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)</div>
  <div class="math-annotated-code-line">            b_logprobs = logprobs.reshape(-1)</div>
  <div class="math-annotated-code-line">            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)</div>
  <div class="math-annotated-code-line">            b_advantages = advantages.reshape(-1)</div>
  <div class="math-annotated-code-line">            b_returns = returns.reshape(-1)</div>
  <div class="math-annotated-code-line">            b_values = values.reshape(-1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            b_inds = np.arange(self.batch_size)</div>
  <div class="math-annotated-code-line">            clipfracs = []</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # Track last minibatch values for logging parity with reference implementation.</div>
  <div class="math-annotated-code-line">            pg_loss = torch.tensor(0.0, device=self.device)</div>
  <div class="math-annotated-code-line">            v_loss = torch.tensor(0.0, device=self.device)</div>
  <div class="math-annotated-code-line">            entropy_loss = torch.tensor(0.0, device=self.device)</div>
  <div class="math-annotated-code-line">            old_approx_kl = torch.tensor(0.0, device=self.device)</div>
  <div class="math-annotated-code-line">            approx_kl = torch.tensor(0.0, device=self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            for epoch in range(self.update_epochs):</div>
  <div class="math-annotated-code-line">                np.random.shuffle(b_inds)</div>
  <div class="math-annotated-code-line">                for start in range(0, self.batch_size, self.minibatch_size):</div>
  <div class="math-annotated-code-line">                    end = start + self.minibatch_size</div>
  <div class="math-annotated-code-line">                    mb_inds = b_inds[start:end]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(</div>
  <div class="math-annotated-code-line">                        b_obs[mb_inds], b_actions[mb_inds]</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\log r_t = \log \pi_{\theta}(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">                    logratio = newlogprob - b_logprobs[mb_inds]</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
r_t = \exp(\log r_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">                    ratio = logratio.exp()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    with torch.no_grad():</div>
  <div class="math-annotated-code-line">                        # calculate approx_kl http://joschu.net/blog/kl-approx.html</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\widehat{D}_{KL}^{old} \approx \mathbb{E}[-\log r_t]
\]</div>
  </div>
  <div class="math-annotated-code-line">                        old_approx_kl = (-logratio).mean()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\widehat{D}_{KL} \approx \mathbb{E}[r_t - 1 - \log r_t]
\]</div>
  </div>
  <div class="math-annotated-code-line">                        approx_kl = ((ratio - 1) - logratio).mean()</div>
  <div class="math-annotated-code-line">                        clipfracs += [</div>
  <div class="math-annotated-code-line">                            ((ratio - 1.0).abs() &gt; self.clip_coef).float().mean().item()</div>
  <div class="math-annotated-code-line">                        ]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    mb_advantages = b_advantages[mb_inds]</div>
  <div class="math-annotated-code-line">                    if self.norm_adv:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
\]</div>
  </div>
  <div class="math-annotated-code-line">                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (</div>
  <div class="math-annotated-code-line">                            mb_advantages.std() + 1e-8</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{pg}^{(1)} = -\hat{A}_t r_t
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    pg_loss1 = -mb_advantages * ratio</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{pg}^{(2)} = -\hat{A}_t \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)
\]</div>
  </div>
  <div class="math-annotated-code-line">                    pg_loss2 = -mb_advantages * torch.clamp(</div>
  <div class="math-annotated-code-line">                        ratio, 1 - self.clip_coef, 1 + self.clip_coef</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{pg} = \mathbb{E}\left[\max\left(\mathcal{L}_{pg}^{(1)}, \mathcal{L}_{pg}^{(2)}\right)\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    newvalue = newvalue.view(-1)</div>
  <div class="math-annotated-code-line">                    if self.clip_vloss:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{V}^{unclip} = (V_{\theta}(s_t)-R_t)^2
\]</div>
  </div>
  <div class="math-annotated-code-line">                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
V_{\theta}^{clip}(s_t) = V_{\theta_{old}}(s_t) + \operatorname{clip}(V_{\theta}(s_t)-V_{\theta_{old}}(s_t), -\epsilon, \epsilon)
\]</div>
  </div>
  <div class="math-annotated-code-line">                        v_clipped = b_values[mb_inds] + torch.clamp(</div>
  <div class="math-annotated-code-line">                            newvalue - b_values[mb_inds],</div>
  <div class="math-annotated-code-line">                            -self.clip_coef,</div>
  <div class="math-annotated-code-line">                            self.clip_coef,</div>
  <div class="math-annotated-code-line">                        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{V}^{clip} = (V_{\theta}^{clip}(s_t)-R_t)^2
\]</div>
  </div>
  <div class="math-annotated-code-line">                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2</div>
  <div class="math-annotated-code-line">                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[\max(\mathcal{L}_{V}^{unclip}, \mathcal{L}_{V}^{clip})\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">                        v_loss = 0.5 * v_loss_max.mean()</div>
  <div class="math-annotated-code-line">                    else:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\theta}(s_t)-R_t)^2\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    entropy_loss = entropy.mean()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L} = \mathcal{L}_{pg} - c_H \mathcal{H} + c_V \mathcal{L}_{V}
\]</div>
  </div>
  <div class="math-annotated-code-line">                    loss = (</div>
  <div class="math-annotated-code-line">                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                    self.optimizer.zero_grad()</div>
  <div class="math-annotated-code-line">                    loss.backward()</div>
  <div class="math-annotated-code-line">                    nn.utils.clip_grad_norm_(</div>
  <div class="math-annotated-code-line">                        self.agent.parameters(), self.max_grad_norm</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                    self.optimizer.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                if self.target_kl is not None:</div>
  <div class="math-annotated-code-line">                    if approx_kl &gt; self.target_kl:</div>
  <div class="math-annotated-code-line">                        break</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()</div>
  <div class="math-annotated-code-line">            var_y = np.var(y_true)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\operatorname{EV} = 1 - \frac{\operatorname{Var}[R - V]}{\operatorname{Var}[R]}
\]</div>
  </div>
  <div class="math-annotated-code-line">            explained_var = (</div>
  <div class="math-annotated-code-line">                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            sps = int(global_step / max(time.time() - start_time, 1e-8))</div>
  <div class="math-annotated-code-line">            log_wandb(</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;charts/learning_rate&quot;: self.optimizer.param_groups[0][&quot;lr&quot;],</div>
  <div class="math-annotated-code-line">                    &quot;losses/value_loss&quot;: v_loss.item(),</div>
  <div class="math-annotated-code-line">                    &quot;losses/policy_loss&quot;: pg_loss.item(),</div>
  <div class="math-annotated-code-line">                    &quot;losses/entropy&quot;: entropy_loss.item(),</div>
  <div class="math-annotated-code-line">                    &quot;losses/old_approx_kl&quot;: old_approx_kl.item(),</div>
  <div class="math-annotated-code-line">                    &quot;losses/approx_kl&quot;: approx_kl.item(),</div>
  <div class="math-annotated-code-line">                    &quot;losses/clipfrac&quot;: float(np.mean(clipfracs)) if clipfracs else 0.0,</div>
  <div class="math-annotated-code-line">                    &quot;losses/explained_variance&quot;: float(explained_var),</div>
  <div class="math-annotated-code-line">                    &quot;charts/SPS&quot;: float(sps),</div>
  <div class="math-annotated-code-line">                },</div>
  <div class="math-annotated-code-line">                step=global_step,</div>
  <div class="math-annotated-code-line">                silent=True,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            print(&quot;SPS:&quot;, sps)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.envs.close()</div>
  <div class="math-annotated-code-line">        self.eval_envs.close()</div>
</div>
