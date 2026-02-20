# `trainers.mpo.agent` Math-Annotated Source

_Source: `minerva/trainers/mpo/agent.py`_

The full file is rendered in one continuous code-style block, with each `# LaTeX:` marker replaced inline by a rendered formula.

## Annotated Source

<div class="math-annotated-codeblock">
  <div class="math-annotated-code-line">from __future__ import annotations</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import copy</div>
  <div class="math-annotated-code-line">import math</div>
  <div class="math-annotated-code-line">from typing import Tuple</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import numpy as np</div>
  <div class="math-annotated-code-line">import torch</div>
  <div class="math-annotated-code-line">import torch.nn as nn</div>
  <div class="math-annotated-code-line">import torch.nn.functional as F</div>
  <div class="math-annotated-code-line">from torch.distributions import Normal</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class LayerNormMLP(nn.Module):</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        in_dim: int,</div>
  <div class="math-annotated-code-line">        layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        activate_final: bool = False,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        super().__init__()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        layers = []</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # First layer: Linear → LayerNorm → tanh</div>
  <div class="math-annotated-code-line">        layers.append(nn.Linear(in_dim, layer_sizes[0]))</div>
  <div class="math-annotated-code-line">        layers.append(nn.LayerNorm(layer_sizes[0]))</div>
  <div class="math-annotated-code-line">        layers.append(nn.Tanh())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Remaining layers: ELU</div>
  <div class="math-annotated-code-line">        for i in range(1, len(layer_sizes)):</div>
  <div class="math-annotated-code-line">            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))</div>
  <div class="math-annotated-code-line">            if activate_final or i &lt; len(layer_sizes) - 1:</div>
  <div class="math-annotated-code-line">                layers.append(nn.ELU())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.net = nn.Sequential(*layers)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        return self.net(x)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class SmallInitLinear(nn.Linear):</div>
  <div class="math-annotated-code-line">    def __init__(self, in_features: int, out_features: int, std: float = 0.01):</div>
  <div class="math-annotated-code-line">        super().__init__(in_features, out_features)</div>
  <div class="math-annotated-code-line">        nn.init.trunc_normal_(self.weight, std=std)</div>
  <div class="math-annotated-code-line">        nn.init.zeros_(self.bias)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class Critic(nn.Module):</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs_dim: int,</div>
  <div class="math-annotated-code-line">        act_dim: int,</div>
  <div class="math-annotated-code-line">        layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        action_low: np.ndarray,</div>
  <div class="math-annotated-code-line">        action_high: np.ndarray,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        super().__init__()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.action_low: torch.Tensor</div>
  <div class="math-annotated-code-line">        self.action_high: torch.Tensor</div>
  <div class="math-annotated-code-line">        self.register_buffer(</div>
  <div class="math-annotated-code-line">            &quot;action_low&quot;, torch.tensor(action_low, dtype=torch.float32)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.register_buffer(</div>
  <div class="math-annotated-code-line">            &quot;action_high&quot;, torch.tensor(action_high, dtype=torch.float32)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.encoder = LayerNormMLP(</div>
  <div class="math-annotated-code-line">            obs_dim + act_dim,</div>
  <div class="math-annotated-code-line">            layer_sizes,</div>
  <div class="math-annotated-code-line">            activate_final=True,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Acme uses a small-init linear head for nondistributional critics.</div>
  <div class="math-annotated-code-line">        self.head = SmallInitLinear(layer_sizes[-1], 1, std=0.01)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def forward(self, obs: torch.Tensor, act: torch.Tensor) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        act = torch.maximum(torch.minimum(act, self.action_high), self.action_low)</div>
  <div class="math-annotated-code-line">        x = torch.cat([obs, act], dim=-1)</div>
  <div class="math-annotated-code-line">        return self.head(self.encoder(x))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class DiagonalGaussianPolicy(nn.Module):</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs_dim: int,</div>
  <div class="math-annotated-code-line">        act_dim: int,</div>
  <div class="math-annotated-code-line">        layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        action_low: np.ndarray | None = None,</div>
  <div class="math-annotated-code-line">        action_high: np.ndarray | None = None,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        super().__init__()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.encoder = LayerNormMLP(</div>
  <div class="math-annotated-code-line">            obs_dim,</div>
  <div class="math-annotated-code-line">            layer_sizes,</div>
  <div class="math-annotated-code-line">            activate_final=True,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.policy_mean = nn.Linear(layer_sizes[-1], act_dim)</div>
  <div class="math-annotated-code-line">        self.policy_logstd = nn.Linear(layer_sizes[-1], act_dim)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        nn.init.kaiming_normal_(</div>
  <div class="math-annotated-code-line">            self.policy_mean.weight, a=0.0, mode=&quot;fan_in&quot;, nonlinearity=&quot;linear&quot;</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        nn.init.zeros_(self.policy_mean.bias)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if action_low is None or action_high is None:</div>
  <div class="math-annotated-code-line">            action_low = -np.ones(act_dim, dtype=np.float32)</div>
  <div class="math-annotated-code-line">            action_high = np.ones(act_dim, dtype=np.float32)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        action_low_t = torch.tensor(action_low, dtype=torch.float32)</div>
  <div class="math-annotated-code-line">        action_high_t = torch.tensor(action_high, dtype=torch.float32)</div>
  <div class="math-annotated-code-line">        self.action_low: torch.Tensor</div>
  <div class="math-annotated-code-line">        self.action_high: torch.Tensor</div>
  <div class="math-annotated-code-line">        self.register_buffer(&quot;action_low&quot;, action_low_t)</div>
  <div class="math-annotated-code-line">        self.register_buffer(&quot;action_high&quot;, action_high_t)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def forward(self, obs: torch.Tensor) -&gt; Tuple[torch.Tensor, torch.Tensor]:</div>
  <div class="math-annotated-code-line">        h = self.encoder(obs)</div>
  <div class="math-annotated-code-line">        return self.forward_with_features(h)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def forward_with_features(</div>
  <div class="math-annotated-code-line">        self, features: torch.Tensor</div>
  <div class="math-annotated-code-line">    ) -&gt; Tuple[torch.Tensor, torch.Tensor]:</div>
  <div class="math-annotated-code-line">        h = features</div>
  <div class="math-annotated-code-line">        mean = self.policy_mean(h)</div>
  <div class="math-annotated-code-line">        log_std = self.policy_logstd(h)</div>
  <div class="math-annotated-code-line">        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)</div>
  <div class="math-annotated-code-line">        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)</div>
  <div class="math-annotated-code-line">        log_std = torch.clamp(log_std, -20.0, 2.0)</div>
  <div class="math-annotated-code-line">        return mean, log_std</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def log_prob(</div>
  <div class="math-annotated-code-line">        self, mean: torch.Tensor, log_std: torch.Tensor, actions_raw: torch.Tensor</div>
  <div class="math-annotated-code-line">    ) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        log_std = torch.clamp(log_std, -20.0, 2.0)</div>
  <div class="math-annotated-code-line">        std = log_std.exp()</div>
  <div class="math-annotated-code-line">        normal = Normal(mean, std)</div>
  <div class="math-annotated-code-line">        log_prob = normal.log_prob(actions_raw)</div>
  <div class="math-annotated-code-line">        return log_prob.sum(dim=-1, keepdim=True)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _clip_to_env_bounds(self, actions_raw: torch.Tensor) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        return torch.maximum(</div>
  <div class="math-annotated-code-line">            torch.minimum(actions_raw, self.action_high), self.action_low</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def sample_action_raw_and_exec(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        mean: torch.Tensor,</div>
  <div class="math-annotated-code-line">        log_std: torch.Tensor,</div>
  <div class="math-annotated-code-line">        deterministic: bool,</div>
  <div class="math-annotated-code-line">        **kwargs,</div>
  <div class="math-annotated-code-line">    ) -&gt; tuple[torch.Tensor, torch.Tensor]:</div>
  <div class="math-annotated-code-line">        log_std = torch.clamp(log_std, -20.0, 2.0)</div>
  <div class="math-annotated-code-line">        if deterministic:</div>
  <div class="math-annotated-code-line">            actions_raw = mean</div>
  <div class="math-annotated-code-line">        else:</div>
  <div class="math-annotated-code-line">            std = log_std.exp()</div>
  <div class="math-annotated-code-line">            normal = Normal(mean, std)</div>
  <div class="math-annotated-code-line">            actions_raw = normal.rsample()</div>
  <div class="math-annotated-code-line">        actions_exec = self._clip_to_env_bounds(actions_raw)</div>
  <div class="math-annotated-code-line">        return actions_raw, actions_exec</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def sample_action(</div>
  <div class="math-annotated-code-line">        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool, **kwargs</div>
  <div class="math-annotated-code-line">    ) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        _, actions_exec = self.sample_action_raw_and_exec(</div>
  <div class="math-annotated-code-line">            mean, log_std, deterministic, **kwargs</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        return actions_exec</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def sample_actions_raw_and_exec(</div>
  <div class="math-annotated-code-line">        self, obs: torch.Tensor, num_actions: int</div>
  <div class="math-annotated-code-line">    ) -&gt; tuple[torch.Tensor, torch.Tensor]:</div>
  <div class="math-annotated-code-line">        mean, log_std = self.forward(obs)</div>
  <div class="math-annotated-code-line">        log_std = torch.clamp(log_std, -20.0, 2.0)</div>
  <div class="math-annotated-code-line">        std = log_std.exp()</div>
  <div class="math-annotated-code-line">        normal = Normal(mean, std)</div>
  <div class="math-annotated-code-line">        actions_raw = normal.rsample(sample_shape=(num_actions,))</div>
  <div class="math-annotated-code-line">        actions_raw = actions_raw.permute(1, 0, 2)</div>
  <div class="math-annotated-code-line">        actions_exec = self._clip_to_env_bounds(actions_raw)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        return actions_raw, actions_exec</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def sample_actions(self, obs: torch.Tensor, num_actions: int) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        _, actions_exec = self.sample_actions_raw_and_exec(obs, num_actions)</div>
  <div class="math-annotated-code-line">        return actions_exec</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def head_parameters(self):</div>
  <div class="math-annotated-code-line">        return list(self.policy_mean.parameters()) + list(</div>
  <div class="math-annotated-code-line">            self.policy_logstd.parameters()</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class MPOAgent:</div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs_dim: int,</div>
  <div class="math-annotated-code-line">        act_dim: int,</div>
  <div class="math-annotated-code-line">        action_low: np.ndarray,</div>
  <div class="math-annotated-code-line">        action_high: np.ndarray,</div>
  <div class="math-annotated-code-line">        device: torch.device,</div>
  <div class="math-annotated-code-line">        policy_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        critic_layer_sizes: Tuple[int, ...],</div>
  <div class="math-annotated-code-line">        gamma: float = 0.99,</div>
  <div class="math-annotated-code-line">        target_networks_update_period: int = 100,</div>
  <div class="math-annotated-code-line">        policy_lr: float = 3e-4,</div>
  <div class="math-annotated-code-line">        q_lr: float = 3e-4,</div>
  <div class="math-annotated-code-line">        kl_epsilon: float = 0.1,</div>
  <div class="math-annotated-code-line">        mstep_kl_epsilon: float = 0.1,</div>
  <div class="math-annotated-code-line">        temperature_init: float = 1.0,</div>
  <div class="math-annotated-code-line">        temperature_lr: float = 3e-4,</div>
  <div class="math-annotated-code-line">        lambda_init: float = 1.0,</div>
  <div class="math-annotated-code-line">        lambda_lr: float = 3e-4,</div>
  <div class="math-annotated-code-line">        epsilon_penalty: float = 0.001,</div>
  <div class="math-annotated-code-line">        max_grad_norm: float = 1.0,</div>
  <div class="math-annotated-code-line">        action_samples: int = 20,</div>
  <div class="math-annotated-code-line">        use_retrace: bool = False,</div>
  <div class="math-annotated-code-line">        retrace_steps: int = 2,</div>
  <div class="math-annotated-code-line">        retrace_mc_actions: int = 8,</div>
  <div class="math-annotated-code-line">        retrace_lambda: float = 0.95,</div>
  <div class="math-annotated-code-line">        optimizer_type: str = &quot;adam&quot;,</div>
  <div class="math-annotated-code-line">        sgd_momentum: float = 0.9,</div>
  <div class="math-annotated-code-line">        init_log_alpha_mean: float = 10.0,</div>
  <div class="math-annotated-code-line">        init_log_alpha_stddev: float = 1000.0,</div>
  <div class="math-annotated-code-line">        m_steps: int = 1,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        self.device = device</div>
  <div class="math-annotated-code-line">        self.gamma = float(gamma)</div>
  <div class="math-annotated-code-line">        self.target_networks_update_period = int(target_networks_update_period)</div>
  <div class="math-annotated-code-line">        self.policy_lr = float(policy_lr)</div>
  <div class="math-annotated-code-line">        self.q_lr = float(q_lr)</div>
  <div class="math-annotated-code-line">        self.kl_epsilon = float(kl_epsilon)</div>
  <div class="math-annotated-code-line">        self.mstep_kl_epsilon = float(mstep_kl_epsilon)</div>
  <div class="math-annotated-code-line">        self.temperature_init = float(temperature_init)</div>
  <div class="math-annotated-code-line">        self.temperature_lr = float(temperature_lr)</div>
  <div class="math-annotated-code-line">        self.lambda_init = float(lambda_init)</div>
  <div class="math-annotated-code-line">        self.lambda_lr = float(lambda_lr)</div>
  <div class="math-annotated-code-line">        self.epsilon_penalty = float(epsilon_penalty)</div>
  <div class="math-annotated-code-line">        self.max_grad_norm = float(max_grad_norm)</div>
  <div class="math-annotated-code-line">        self.action_samples = int(action_samples)</div>
  <div class="math-annotated-code-line">        self.use_retrace = bool(use_retrace)</div>
  <div class="math-annotated-code-line">        self.retrace_steps = int(retrace_steps)</div>
  <div class="math-annotated-code-line">        self.retrace_mc_actions = int(retrace_mc_actions)</div>
  <div class="math-annotated-code-line">        self.retrace_lambda = float(retrace_lambda)</div>
  <div class="math-annotated-code-line">        self.optimizer_type = str(optimizer_type)</div>
  <div class="math-annotated-code-line">        self.sgd_momentum = float(sgd_momentum)</div>
  <div class="math-annotated-code-line">        self.m_steps = int(m_steps)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Learner step counter for periodic target synchronization.</div>
  <div class="math-annotated-code-line">        self._num_steps = 0</div>
  <div class="math-annotated-code-line">        self._skipped_nonfinite_batches = 0</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.policy = DiagonalGaussianPolicy(</div>
  <div class="math-annotated-code-line">            obs_dim,</div>
  <div class="math-annotated-code-line">            act_dim,</div>
  <div class="math-annotated-code-line">            layer_sizes=policy_layer_sizes,</div>
  <div class="math-annotated-code-line">            action_low=action_low,</div>
  <div class="math-annotated-code-line">            action_high=action_high,</div>
  <div class="math-annotated-code-line">        ).to(device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.policy_target = copy.deepcopy(self.policy).to(device)</div>
  <div class="math-annotated-code-line">        self.policy_target.eval()</div>
  <div class="math-annotated-code-line">        # Single critic using the Acme control-network critic torso.</div>
  <div class="math-annotated-code-line">        self.q = Critic(</div>
  <div class="math-annotated-code-line">            obs_dim,</div>
  <div class="math-annotated-code-line">            act_dim,</div>
  <div class="math-annotated-code-line">            layer_sizes=critic_layer_sizes,</div>
  <div class="math-annotated-code-line">            action_low=action_low,</div>
  <div class="math-annotated-code-line">            action_high=action_high,</div>
  <div class="math-annotated-code-line">        ).to(device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        print(self.policy)</div>
  <div class="math-annotated-code-line">        print(self.q)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.q_target = copy.deepcopy(self.q).to(device)</div>
  <div class="math-annotated-code-line">        self.q_target.eval()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Train policy encoder + head together; critics share a separate encoder.</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tilde{\lambda}_{\pi} = \frac{\lambda_{\pi}}{\max(1, M)}
\]</div>
  </div>
  <div class="math-annotated-code-line">        policy_lr_effective = float(self.policy_lr) / max(1, int(self.m_steps))</div>
  <div class="math-annotated-code-line">        self.policy_opt = self._build_optimizer(</div>
  <div class="math-annotated-code-line">            self.policy.parameters(), lr=policy_lr_effective</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Critic optimizer.</div>
  <div class="math-annotated-code-line">        self.q_opt = self._build_optimizer(self.q.parameters(), lr=self.q_lr)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Dual variables (temperature + KL multipliers) in log-space.</div>
  <div class="math-annotated-code-line">        self.log_temperature = nn.Parameter(</div>
  <div class="math-annotated-code-line">            torch.tensor(self.temperature_init, device=device)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        lambda_init_t = torch.tensor(self.lambda_init, device=device)</div>
  <div class="math-annotated-code-line">        lambda_init_t = torch.clamp(lambda_init_t, min=1e-8)</div>
  <div class="math-annotated-code-line">        dual_shape = (act_dim,)</div>
  <div class="math-annotated-code-line">        self.log_alpha_mean = nn.Parameter(</div>
  <div class="math-annotated-code-line">            torch.full(dual_shape, init_log_alpha_mean, device=device)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.log_alpha_stddev = nn.Parameter(</div>
  <div class="math-annotated-code-line">            torch.full(dual_shape, init_log_alpha_stddev, device=device)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Dual optimizer uses separate LRs for temperature vs alphas.</div>
  <div class="math-annotated-code-line">        temperature_params = [self.log_temperature]</div>
  <div class="math-annotated-code-line">        alpha_params = [self.log_alpha_mean, self.log_alpha_stddev]</div>
  <div class="math-annotated-code-line">        self.dual_opt = self._build_optimizer(</div>
  <div class="math-annotated-code-line">            [</div>
  <div class="math-annotated-code-line">                {&quot;params&quot;: temperature_params, &quot;lr&quot;: self.temperature_lr},</div>
  <div class="math-annotated-code-line">                {&quot;params&quot;: alpha_params, &quot;lr&quot;: self.lambda_lr},</div>
  <div class="math-annotated-code-line">            ],</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _build_optimizer(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        params,</div>
  <div class="math-annotated-code-line">        lr: float | None = None,</div>
  <div class="math-annotated-code-line">    ) -&gt; torch.optim.Optimizer:</div>
  <div class="math-annotated-code-line">        optimizer_type = self.optimizer_type.strip().lower()</div>
  <div class="math-annotated-code-line">        kwargs: dict[str, float] = {}</div>
  <div class="math-annotated-code-line">        if lr is not None:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;lr&quot;] = float(lr)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if optimizer_type == &quot;sgd&quot;:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;momentum&quot;] = float(self.sgd_momentum)</div>
  <div class="math-annotated-code-line">            return torch.optim.SGD(params, **kwargs)</div>
  <div class="math-annotated-code-line">        else:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;eps&quot;] = 1e-5</div>
  <div class="math-annotated-code-line">            return torch.optim.Adam(params, **kwargs)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _kl_diag_gaussian_per_dim(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        mean_p,</div>
  <div class="math-annotated-code-line">        log_std_p,</div>
  <div class="math-annotated-code-line">        mean_q,</div>
  <div class="math-annotated-code-line">        log_std_q,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        inv_var_q = torch.exp(-2.0 * log_std_q)</div>
  <div class="math-annotated-code-line">        var_ratio = torch.exp(2.0 * (log_std_p - log_std_q))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        return 0.5 * (var_ratio + (mean_p - mean_q).pow(2) * inv_var_q - 1.0) + (</div>
  <div class="math-annotated-code-line">            log_std_q - log_std_p</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _compute_weights_and_temperature_loss(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        q_values: torch.Tensor,</div>
  <div class="math-annotated-code-line">        epsilon: float,</div>
  <div class="math-annotated-code-line">        temperature: torch.Tensor,</div>
  <div class="math-annotated-code-line">    ) -&gt; tuple[torch.Tensor, torch.Tensor]:</div>
  <div class="math-annotated-code-line">        &quot;&quot;&quot;Acme-style E-step weights and dual temperature loss.</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        q_values shape (B,N). Returns weights (B,N) detached and loss scalar.</div>
  <div class="math-annotated-code-line">        &quot;&quot;&quot;</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\bar{Q}_{b,i} = \frac{Q_{b,i}}{\eta}
\]</div>
  </div>
  <div class="math-annotated-code-line">        q_detached = q_values.detach() / temperature</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
q_{b,i} = \frac{\exp(\bar{Q}_{b,i})}{\sum_j \exp(\bar{Q}_{b,j})}
\]</div>
  </div>
  <div class="math-annotated-code-line">        weights = torch.softmax(q_detached, dim=1).detach()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\log Z_b = \log \sum_i \exp(\bar{Q}_{b,i})
\]</div>
  </div>
  <div class="math-annotated-code-line">        q_logsumexp = torch.logsumexp(q_detached, dim=1)</div>
  <div class="math-annotated-code-line">        log_num_actions = math.log(q_values.shape[1])</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\eta} = \eta\left(\epsilon + \frac{1}{B}\sum_b \log Z_b - \log N\right)
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_temperature = temperature * (</div>
  <div class="math-annotated-code-line">            float(epsilon) + q_logsumexp.mean() - log_num_actions</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        return weights, loss_temperature</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _compute_nonparametric_kl_from_weights(</div>
  <div class="math-annotated-code-line">        self, weights: torch.Tensor</div>
  <div class="math-annotated-code-line">    ) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        &quot;&quot;&quot;Estimates KL(nonparametric || target) like Acme&#x27;s diagnostics.</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        weights shape (B,N). Returns (B,) KL.</div>
  <div class="math-annotated-code-line">        &quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">        n = float(weights.shape[1])</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\ell_{b,i} = \log\!\left(N q_{b,i}\right)
\]</div>
  </div>
  <div class="math-annotated-code-line">        integrand = torch.log(n * weights + 1e-8)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
D_{KL}(q_b\|u) = \sum_i q_{b,i}\log\!\left(N q_{b,i}\right)
\]</div>
  </div>
  <div class="math-annotated-code-line">        return (weights * integrand).sum(dim=1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _to_device_tensor(self, value: np.ndarray | torch.Tensor) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        &quot;&quot;&quot;Fast path for float32 host arrays -&gt; device tensors.&quot;&quot;&quot;</div>
  <div class="math-annotated-code-line">        if isinstance(value, torch.Tensor):</div>
  <div class="math-annotated-code-line">            return value.to(</div>
  <div class="math-annotated-code-line">                device=self.device,</div>
  <div class="math-annotated-code-line">                dtype=torch.float32,</div>
  <div class="math-annotated-code-line">                non_blocking=True,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        arr = np.asarray(value, dtype=np.float32)</div>
  <div class="math-annotated-code-line">        if not arr.flags.c_contiguous:</div>
  <div class="math-annotated-code-line">            arr = np.ascontiguousarray(arr)</div>
  <div class="math-annotated-code-line">        return torch.from_numpy(arr).to(device=self.device, non_blocking=True)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _assert_finite_tensors(self, tensors: dict[str, torch.Tensor]) -&gt; bool:</div>
  <div class="math-annotated-code-line">        try:</div>
  <div class="math-annotated-code-line">            for name, tensor in tensors.items():</div>
  <div class="math-annotated-code-line">                assert bool(</div>
  <div class="math-annotated-code-line">                    torch.isfinite(tensor).all()</div>
  <div class="math-annotated-code-line">                ), f&quot;non-finite values in &#x27;{name}&#x27;&quot;</div>
  <div class="math-annotated-code-line">        except AssertionError as exc:</div>
  <div class="math-annotated-code-line">            self._skipped_nonfinite_batches += 1</div>
  <div class="math-annotated-code-line">            if (</div>
  <div class="math-annotated-code-line">                self._skipped_nonfinite_batches &lt;= 5</div>
  <div class="math-annotated-code-line">                or self._skipped_nonfinite_batches % 100 == 0</div>
  <div class="math-annotated-code-line">            ):</div>
  <div class="math-annotated-code-line">                print(</div>
  <div class="math-annotated-code-line">                    &quot;[MPO][warn] &quot;</div>
  <div class="math-annotated-code-line">                    f&quot;{exc}; skipped batch #{self._skipped_nonfinite_batches}&quot;</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">            return False</div>
  <div class="math-annotated-code-line">        return True</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def act_with_logp(</div>
  <div class="math-annotated-code-line">        self, obs: np.ndarray, deterministic: bool = False</div>
  <div class="math-annotated-code-line">    ) -&gt; tuple[np.ndarray, np.ndarray, float]:</div>
  <div class="math-annotated-code-line">        obs_t = self._to_device_tensor(obs).unsqueeze(0)</div>
  <div class="math-annotated-code-line">        with torch.inference_mode():</div>
  <div class="math-annotated-code-line">            mean, log_std = self.policy(obs_t)</div>
  <div class="math-annotated-code-line">            action_raw, action_exec = self.policy.sample_action_raw_and_exec(</div>
  <div class="math-annotated-code-line">                mean, log_std, deterministic</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            logp = self.policy.log_prob(mean, log_std, action_raw)</div>
  <div class="math-annotated-code-line">        return (</div>
  <div class="math-annotated-code-line">            action_exec.cpu().numpy().squeeze(0),</div>
  <div class="math-annotated-code-line">            action_raw.cpu().numpy().squeeze(0),</div>
  <div class="math-annotated-code-line">            float(logp.item()),</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def act(self, obs: np.ndarray, deterministic: bool = False) -&gt; np.ndarray:</div>
  <div class="math-annotated-code-line">        action_exec, _, _ = self.act_with_logp(obs, deterministic=deterministic)</div>
  <div class="math-annotated-code-line">        return action_exec</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _expected_q_current(self, obs: torch.Tensor) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            actions = self.policy_target.sample_actions(</div>
  <div class="math-annotated-code-line">                obs, num_actions=self.retrace_mc_actions</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            batch_size = obs.shape[0]</div>
  <div class="math-annotated-code-line">            obs_rep = obs.unsqueeze(1).expand(</div>
  <div class="math-annotated-code-line">                batch_size, self.retrace_mc_actions, obs.shape[-1]</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            obs_flat = obs_rep.reshape(-1, obs.shape[-1])</div>
  <div class="math-annotated-code-line">            act_flat = actions.reshape(-1, actions.shape[-1])</div>
  <div class="math-annotated-code-line">            q = self.q_target(obs_flat, act_flat)</div>
  <div class="math-annotated-code-line">            return q.reshape(batch_size, self.retrace_mc_actions).mean(</div>
  <div class="math-annotated-code-line">                dim=1, keepdim=True</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _retrace_q_target(self, batch: dict) -&gt; torch.Tensor:</div>
  <div class="math-annotated-code-line">        obs_seq = self._to_device_tensor(batch[&quot;obs&quot;])</div>
  <div class="math-annotated-code-line">        actions_exec_seq = self._to_device_tensor(batch[&quot;actions_exec&quot;])</div>
  <div class="math-annotated-code-line">        actions_raw_seq = self._to_device_tensor(batch[&quot;actions_raw&quot;])</div>
  <div class="math-annotated-code-line">        rewards_seq = self._to_device_tensor(batch[&quot;rewards&quot;])</div>
  <div class="math-annotated-code-line">        next_obs_seq = self._to_device_tensor(batch[&quot;next_obs&quot;])</div>
  <div class="math-annotated-code-line">        dones_seq = self._to_device_tensor(batch[&quot;dones&quot;])</div>
  <div class="math-annotated-code-line">        behaviour_logp_seq = self._to_device_tensor(batch[&quot;behaviour_logp&quot;])</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        batch_size, seq_len, obs_dim = obs_seq.shape</div>
  <div class="math-annotated-code-line">        act_dim = actions_exec_seq.shape[-1]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            obs_flat = obs_seq.reshape(batch_size * seq_len, obs_dim)</div>
  <div class="math-annotated-code-line">            act_exec_flat = actions_exec_seq.reshape(batch_size * seq_len, act_dim)</div>
  <div class="math-annotated-code-line">            q_t = self.q_target(obs_flat, act_exec_flat).reshape(batch_size, seq_len, 1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, obs_dim)</div>
  <div class="math-annotated-code-line">            v_next = self._expected_q_current(next_obs_flat).reshape(</div>
  <div class="math-annotated-code-line">                batch_size, seq_len, 1</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\delta_t = r_t + \gamma(1-d_t)V(s_{t+1}) - Q(s_t,a_t)
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            delta = rewards_seq + (1.0 - dones_seq) * self.gamma * v_next - q_t</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            mean, log_std = self.policy_target(obs_flat)</div>
  <div class="math-annotated-code-line">            actions_raw_flat = actions_raw_seq.reshape(batch_size * seq_len, act_dim)</div>
  <div class="math-annotated-code-line">            log_pi = self.policy_target.log_prob(</div>
  <div class="math-annotated-code-line">                mean, log_std, actions_raw_flat</div>
  <div class="math-annotated-code-line">            ).reshape(batch_size, seq_len, 1)</div>
  <div class="math-annotated-code-line">            log_b = behaviour_logp_seq</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\log \rho_t = \log \pi(a_t|s_t) - \log b(a_t|s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">            log_ratio = log_pi - log_b</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\rho_t = \exp(\log \rho_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">            rho = torch.exp(log_ratio).squeeze(-1)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
c_t = \lambda \min(1, \rho_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">            c = (self.retrace_lambda * torch.minimum(torch.ones_like(rho), rho)).detach()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # Correct Retrace recursion:</div>
  <div class="math-annotated-code-line">            # Qret(s0,a0) = Q(s0,a0) + sum_{t=0}^{T-1} gamma^t (prod_{i=1}^t c_i) delta_t</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
Q^{ret} \leftarrow Q(s_0, a_0)
\]</div>
  </div>
  <div class="math-annotated-code-line">            q_ret = q_t[:, 0, :].clone()</div>
  <div class="math-annotated-code-line">            cont = torch.ones((batch_size, 1), device=self.device)</div>
  <div class="math-annotated-code-line">            c_prod = torch.ones((batch_size, 1), device=self.device)</div>
  <div class="math-annotated-code-line">            discount = torch.ones((batch_size, 1), device=self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            dones_flat = dones_seq.squeeze(-1)  # (B,T)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            for t in range(seq_len):</div>
  <div class="math-annotated-code-line">                if t &gt; 0:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_t \leftarrow m_{t-1}(1-d_{t-1})
\]</div>
  </div>
  <div class="math-annotated-code-line">                    cont = cont * (1.0 - dones_flat[:, t - 1 : t])</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
C_t \leftarrow C_{t-1} c_t
\]</div>
  </div>
  <div class="math-annotated-code-line">                    c_prod = c_prod * c[:, t : t + 1]</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\Gamma_t \leftarrow \Gamma_{t-1}\gamma
\]</div>
  </div>
  <div class="math-annotated-code-line">                    discount = discount * self.gamma</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
Q^{ret} \leftarrow Q^{ret} + m_t \Gamma_t C_t \delta_t
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">                q_ret = q_ret + cont * discount * c_prod * delta[:, t, :]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        return q_ret</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def update(self, batch: dict) -&gt; dict | None:</div>
  <div class="math-annotated-code-line">        if self._num_steps % self.target_networks_update_period == 0:</div>
  <div class="math-annotated-code-line">            self.policy_target.load_state_dict(self.policy.state_dict())</div>
  <div class="math-annotated-code-line">            self.q_target.load_state_dict(self.q.state_dict())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs_batch = batch.get(&quot;obs&quot;)</div>
  <div class="math-annotated-code-line">        is_sequence_batch = isinstance(obs_batch, (np.ndarray, torch.Tensor)) and (</div>
  <div class="math-annotated-code-line">            obs_batch.ndim == 3</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        use_retrace = self.use_retrace</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if use_retrace and is_sequence_batch and self.retrace_steps &gt; 1:</div>
  <div class="math-annotated-code-line">            obs_seq = self._to_device_tensor(batch[&quot;obs&quot;])</div>
  <div class="math-annotated-code-line">            actions_exec_seq = self._to_device_tensor(batch[&quot;actions_exec&quot;])</div>
  <div class="math-annotated-code-line">            actions_raw_seq = self._to_device_tensor(batch[&quot;actions_raw&quot;])</div>
  <div class="math-annotated-code-line">            rewards_seq = self._to_device_tensor(batch[&quot;rewards&quot;])</div>
  <div class="math-annotated-code-line">            next_obs_seq = self._to_device_tensor(batch[&quot;next_obs&quot;])</div>
  <div class="math-annotated-code-line">            dones_seq = self._to_device_tensor(batch[&quot;dones&quot;])</div>
  <div class="math-annotated-code-line">            behaviour_logp_seq = self._to_device_tensor(batch[&quot;behaviour_logp&quot;])</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            if not self._assert_finite_tensors(</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;obs&quot;: obs_seq,</div>
  <div class="math-annotated-code-line">                    &quot;actions_exec&quot;: actions_exec_seq,</div>
  <div class="math-annotated-code-line">                    &quot;actions_raw&quot;: actions_raw_seq,</div>
  <div class="math-annotated-code-line">                    &quot;rewards&quot;: rewards_seq,</div>
  <div class="math-annotated-code-line">                    &quot;next_obs&quot;: next_obs_seq,</div>
  <div class="math-annotated-code-line">                    &quot;dones&quot;: dones_seq,</div>
  <div class="math-annotated-code-line">                    &quot;behaviour_logp&quot;: behaviour_logp_seq,</div>
  <div class="math-annotated-code-line">                }</div>
  <div class="math-annotated-code-line">            ):</div>
  <div class="math-annotated-code-line">                return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            target = self._retrace_q_target(</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;obs&quot;: obs_seq,</div>
  <div class="math-annotated-code-line">                    &quot;actions_exec&quot;: actions_exec_seq,</div>
  <div class="math-annotated-code-line">                    &quot;actions_raw&quot;: actions_raw_seq,</div>
  <div class="math-annotated-code-line">                    &quot;rewards&quot;: rewards_seq,</div>
  <div class="math-annotated-code-line">                    &quot;next_obs&quot;: next_obs_seq,</div>
  <div class="math-annotated-code-line">                    &quot;dones&quot;: dones_seq,</div>
  <div class="math-annotated-code-line">                    &quot;behaviour_logp&quot;: behaviour_logp_seq,</div>
  <div class="math-annotated-code-line">                }</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            obs = obs_seq[:, 0, :]</div>
  <div class="math-annotated-code-line">            actions = actions_exec_seq[:, 0, :]</div>
  <div class="math-annotated-code-line">        else:</div>
  <div class="math-annotated-code-line">            obs = self._to_device_tensor(batch[&quot;obs&quot;])</div>
  <div class="math-annotated-code-line">            actions_key = (</div>
  <div class="math-annotated-code-line">                &quot;actions_exec&quot; if &quot;actions_exec&quot; in batch.keys() else &quot;actions&quot;</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            actions = self._to_device_tensor(batch[actions_key])</div>
  <div class="math-annotated-code-line">            rewards = self._to_device_tensor(batch[&quot;rewards&quot;])</div>
  <div class="math-annotated-code-line">            next_obs = self._to_device_tensor(batch[&quot;next_obs&quot;])</div>
  <div class="math-annotated-code-line">            dones = self._to_device_tensor(batch[&quot;dones&quot;])</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            if not self._assert_finite_tensors(</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;obs&quot;: obs,</div>
  <div class="math-annotated-code-line">                    &quot;actions&quot;: actions,</div>
  <div class="math-annotated-code-line">                    &quot;rewards&quot;: rewards,</div>
  <div class="math-annotated-code-line">                    &quot;next_obs&quot;: next_obs,</div>
  <div class="math-annotated-code-line">                    &quot;dones&quot;: dones,</div>
  <div class="math-annotated-code-line">                }</div>
  <div class="math-annotated-code-line">            ):</div>
  <div class="math-annotated-code-line">                return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            with torch.no_grad():</div>
  <div class="math-annotated-code-line">                next_actions = self.policy_target.sample_actions(</div>
  <div class="math-annotated-code-line">                    next_obs, num_actions=self.action_samples</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                batch_size = next_obs.shape[0]</div>
  <div class="math-annotated-code-line">                next_obs_rep = next_obs.unsqueeze(1).expand(</div>
  <div class="math-annotated-code-line">                    batch_size, self.action_samples, next_obs.shape[-1]</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                next_obs_flat = next_obs_rep.reshape(-1, next_obs.shape[-1])</div>
  <div class="math-annotated-code-line">                next_act_flat = next_actions.reshape(-1, next_actions.shape[-1])</div>
  <div class="math-annotated-code-line">                q_target = self.q_target(next_obs_flat, next_act_flat)</div>
  <div class="math-annotated-code-line">                q_target = q_target.reshape(batch_size, self.action_samples).mean(</div>
  <div class="math-annotated-code-line">                    dim=1, keepdim=True</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
y_t = r_t + \gamma(1-d_t)\bar{Q}(s_{t+1})
\]</div>
  </div>
  <div class="math-annotated-code-line">                target = rewards + (1.0 - dones) * self.gamma * q_target</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if not self._assert_finite_tensors(</div>
  <div class="math-annotated-code-line">            {&quot;obs&quot;: obs, &quot;actions&quot;: actions, &quot;target&quot;: target}</div>
  <div class="math-annotated-code-line">        ):</div>
  <div class="math-annotated-code-line">            return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        q = self.q(obs, actions)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_Q = \mathbb{E}\!\left[(Q(s_t,a_t) - y_t)^2\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">        q_loss = F.mse_loss(q, target)</div>
  <div class="math-annotated-code-line">        if not self._assert_finite_tensors({&quot;q&quot;: q, &quot;q_loss&quot;: q_loss}):</div>
  <div class="math-annotated-code-line">            return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Phase A: critic update</div>
  <div class="math-annotated-code-line">        self.q_opt.zero_grad()</div>
  <div class="math-annotated-code-line">        q_loss.backward()</div>
  <div class="math-annotated-code-line">        nn.utils.clip_grad_norm_(</div>
  <div class="math-annotated-code-line">            self.q.parameters(),</div>
  <div class="math-annotated-code-line">            self.max_grad_norm,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.q_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Phase B (Acme-style): update dual vars then do policy M-step.</div>
  <div class="math-annotated-code-line">        batch_size = obs.shape[0]</div>
  <div class="math-annotated-code-line">        num_samples = self.action_samples</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        mean_online, log_std_online = self.policy(obs)</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            mean_target, log_std_target = self.policy_target(obs)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            sampled_actions_raw, sampled_actions_exec = (</div>
  <div class="math-annotated-code-line">                self.policy_target.sample_actions_raw_and_exec(</div>
  <div class="math-annotated-code-line">                    obs, num_actions=num_samples</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">            )  # (B,N,D)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            obs_rep = obs.unsqueeze(1).expand(batch_size, num_samples, obs.shape[-1])</div>
  <div class="math-annotated-code-line">            obs_flat = obs_rep.reshape(-1, obs.shape[-1])</div>
  <div class="math-annotated-code-line">            act_exec_flat = sampled_actions_exec.reshape(</div>
  <div class="math-annotated-code-line">                -1, sampled_actions_exec.shape[-1]</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            q_vals = self.q_target(obs_flat, act_exec_flat).reshape(</div>
  <div class="math-annotated-code-line">                batch_size, num_samples</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if not self._assert_finite_tensors({&quot;q_vals&quot;: q_vals}):</div>
  <div class="math-annotated-code-line">            return None</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        min_log = torch.tensor(-18.0, device=self.device)</div>
  <div class="math-annotated-code-line">        log_temp = torch.maximum(self.log_temperature, min_log)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\eta = \operatorname{softplus}(\tilde{\eta}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">        temperature = F.softplus(log_temp) + 1e-8</div>
  <div class="math-annotated-code-line">        weights, loss_temperature = self._compute_weights_and_temperature_loss(</div>
  <div class="math-annotated-code-line">            q_vals, self.kl_epsilon, temperature</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # KL(nonparametric || target) diagnostic (relative).</div>
  <div class="math-annotated-code-line">        kl_nonparametric = self._compute_nonparametric_kl_from_weights(weights)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathrm{KL}_{rel} = \frac{\mathbb{E}[D_{KL}(q\|u)]}{\epsilon}
\]</div>
  </div>
  <div class="math-annotated-code-line">        kl_q_rel = kl_nonparametric.mean() / float(self.kl_epsilon)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Compute Acme-style decomposed losses.</div>
  <div class="math-annotated-code-line">        std_online = torch.exp(log_std_online)</div>
  <div class="math-annotated-code-line">        std_target = torch.exp(log_std_target)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Fixed distributions for decomposition.</div>
  <div class="math-annotated-code-line">        actions = sampled_actions_raw.detach()  # (B,N,D), stop-gradient wrt sampling.</div>
  <div class="math-annotated-code-line">        mean_online_exp = mean_online.unsqueeze(1)</div>
  <div class="math-annotated-code-line">        std_online_exp = std_online.unsqueeze(1)</div>
  <div class="math-annotated-code-line">        mean_target_exp = mean_target.unsqueeze(1)</div>
  <div class="math-annotated-code-line">        std_target_exp = std_target.unsqueeze(1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # fixed_stddev: mean=online_mean, std=target_std</div>
  <div class="math-annotated-code-line">        log_prob_fixed_stddev = (</div>
  <div class="math-annotated-code-line">            Normal(mean_online_exp, std_target_exp).log_prob(actions).sum(dim=-1)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        # fixed_mean: mean=target_mean, std=online_std</div>
  <div class="math-annotated-code-line">        log_prob_fixed_mean = (</div>
  <div class="math-annotated-code-line">            Normal(mean_target_exp, std_online_exp).log_prob(actions).sum(dim=-1)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Cross entropy / weighted log-prob.</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi,\mu} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu,\sigma&#x27;}(a_{b,i}|s_b)
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_policy_mean = -(weights * log_prob_fixed_stddev).sum(dim=1).mean()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi,\sigma} = -\mathbb{E}_b\sum_i q_{b,i}\log \pi_{\mu&#x27;,\sigma}(a_{b,i}|s_b)
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_policy_std = -(weights * log_prob_fixed_mean).sum(dim=1).mean()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi} = \mathcal{L}_{\pi,\mu} + \mathcal{L}_{\pi,\sigma}
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_policy = loss_policy_mean + loss_policy_std</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        kl_mean = self._kl_diag_gaussian_per_dim(</div>
  <div class="math-annotated-code-line">            mean_target.detach(),</div>
  <div class="math-annotated-code-line">            log_std_target.detach(),</div>
  <div class="math-annotated-code-line">            mean_online,</div>
  <div class="math-annotated-code-line">            log_std_target.detach(),</div>
  <div class="math-annotated-code-line">        )  # (B,D)</div>
  <div class="math-annotated-code-line">        kl_std = self._kl_diag_gaussian_per_dim(</div>
  <div class="math-annotated-code-line">            mean_target.detach(),</div>
  <div class="math-annotated-code-line">            log_std_target.detach(),</div>
  <div class="math-annotated-code-line">            mean_target.detach(),</div>
  <div class="math-annotated-code-line">            log_std_online,</div>
  <div class="math-annotated-code-line">        )  # (B,D)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\bar{D}_{\mu,j} = \frac{1}{B}\sum_b D_{\mu,b,j}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        mean_kl_mean = kl_mean.mean(dim=0)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\bar{D}_{\sigma,j} = \frac{1}{B}\sum_b D_{\sigma,b,j}
\]</div>
  </div>
  <div class="math-annotated-code-line">        mean_kl_std = kl_std.mean(dim=0)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        log_alpha_mean = torch.maximum(self.log_alpha_mean, min_log)</div>
  <div class="math-annotated-code-line">        log_alpha_stddev = torch.maximum(self.log_alpha_stddev, min_log)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\mu,j} = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        alpha_mean = F.softplus(log_alpha_mean) + 1e-8</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\sigma,j} = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">        alpha_std = F.softplus(log_alpha_stddev) + 1e-8</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{KL,\mu} = \sum_j \alpha_{\mu,j}\bar{D}_{\mu,j}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        loss_kl_mean = (alpha_mean.detach() * mean_kl_mean).sum()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{KL,\sigma} = \sum_j \alpha_{\sigma,j}\bar{D}_{\sigma,j}
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_kl_std = (alpha_std.detach() * mean_kl_std).sum()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{KL} = \mathcal{L}_{KL,\mu} + \mathcal{L}_{KL,\sigma}
\]</div>
  </div>
  <div class="math-annotated-code-line">        loss_kl_penalty = loss_kl_mean + loss_kl_std</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        loss_alpha_mean = (</div>
  <div class="math-annotated-code-line">            alpha_mean * (self.mstep_kl_epsilon - mean_kl_mean.detach())</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\alpha_{\mu}} = \sum_j \alpha_{\mu,j}(\epsilon_{KL} - \bar{D}_{\mu,j})
\]</div>
  </div>
  <div class="math-annotated-code-line">        ).sum()</div>
  <div class="math-annotated-code-line">        loss_alpha_std = (</div>
  <div class="math-annotated-code-line">            alpha_std * (self.mstep_kl_epsilon - mean_kl_std.detach())</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\alpha_{\sigma}} = \sum_j \alpha_{\sigma,j}(\epsilon_{KL} - \bar{D}_{\sigma,j})
\]</div>
  </div>
  <div class="math-annotated-code-line">        ).sum()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Update dual variables (temperature + alphas).</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{dual} = \mathcal{L}_{\eta} + \mathcal{L}_{\alpha_{\mu}} + \mathcal{L}_{\alpha_{\sigma}}
\]</div>
  </div>
  <div class="math-annotated-code-line">        dual_loss = loss_temperature + loss_alpha_mean + loss_alpha_std</div>
  <div class="math-annotated-code-line">        self.dual_opt.zero_grad()</div>
  <div class="math-annotated-code-line">        dual_loss.backward()</div>
  <div class="math-annotated-code-line">        nn.utils.clip_grad_norm_(</div>
  <div class="math-annotated-code-line">            [</div>
  <div class="math-annotated-code-line">                p</div>
  <div class="math-annotated-code-line">                for p in [</div>
  <div class="math-annotated-code-line">                    self.log_temperature,</div>
  <div class="math-annotated-code-line">                    self.log_alpha_mean,</div>
  <div class="math-annotated-code-line">                    self.log_alpha_stddev,</div>
  <div class="math-annotated-code-line">                ]</div>
  <div class="math-annotated-code-line">                if p is not None</div>
  <div class="math-annotated-code-line">            ],</div>
  <div class="math-annotated-code-line">            self.max_grad_norm,</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.dual_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # ---------- after dual_opt.step() ----------</div>
  <div class="math-annotated-code-line">        # Recompute the post-update dual multipliers and freeze fixed E-step tensors.</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            mean_target_det = mean_target.detach()</div>
  <div class="math-annotated-code-line">            log_std_target_det = log_std_target.detach()</div>
  <div class="math-annotated-code-line">            std_target_det = std_target.detach()</div>
  <div class="math-annotated-code-line">            weights_det = weights.detach()  # (B,N)</div>
  <div class="math-annotated-code-line">            actions_det = actions.detach()  # (B,N,D)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Recompute dual multipliers AFTER dual_opt.step()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\mu,j}&#x27; = \operatorname{softplus}(\tilde{\alpha}_{\mu,j}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">        alpha_mean_det = (F.softplus(self.log_alpha_mean) + 1e-8).detach()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\sigma,j}&#x27; = \operatorname{softplus}(\tilde{\alpha}_{\sigma,j}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">        alpha_std_det = (F.softplus(self.log_alpha_stddev) + 1e-8).detach()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Parameter delta diagnostic: snapshot BEFORE M-step</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            params_before = (</div>
  <div class="math-annotated-code-line">                nn.utils.parameters_to_vector(self.policy.parameters()).detach().clone()</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Inner M-step (recompute online outputs each iteration)</div>
  <div class="math-annotated-code-line">        for _ in range(int(self.m_steps)):</div>
  <div class="math-annotated-code-line">            mean_online, log_std_online = self.policy(obs)  # (B,D), (B,D)</div>
  <div class="math-annotated-code-line">            std_online = torch.exp(log_std_online)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # expand shapes for (B,N,D)</div>
  <div class="math-annotated-code-line">            mean_online_exp = mean_online.unsqueeze(1)</div>
  <div class="math-annotated-code-line">            std_online_exp = std_online.unsqueeze(1)</div>
  <div class="math-annotated-code-line">            mean_target_exp = mean_target_det.unsqueeze(1)</div>
  <div class="math-annotated-code-line">            std_target_exp = std_target_det.unsqueeze(1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # cross-entropy pieces</div>
  <div class="math-annotated-code-line">            log_prob_fixed_stddev = (</div>
  <div class="math-annotated-code-line">                Normal(mean_online_exp, std_target_exp)</div>
  <div class="math-annotated-code-line">                .log_prob(actions_det)</div>
  <div class="math-annotated-code-line">                .sum(dim=-1)</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            log_prob_fixed_mean = (</div>
  <div class="math-annotated-code-line">                Normal(mean_target_exp, std_online_exp)</div>
  <div class="math-annotated-code-line">                .log_prob(actions_det)</div>
  <div class="math-annotated-code-line">                .sum(dim=-1)</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            loss_policy_mean = -(weights_det * log_prob_fixed_stddev).sum(dim=1).mean()</div>
  <div class="math-annotated-code-line">            loss_policy_std = -(weights_det * log_prob_fixed_mean).sum(dim=1).mean()</div>
  <div class="math-annotated-code-line">            loss_policy = loss_policy_mean + loss_policy_std</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # KL penalties (compare online -&gt; frozen target)</div>
  <div class="math-annotated-code-line">            kl_mean = self._kl_diag_gaussian_per_dim(</div>
  <div class="math-annotated-code-line">                mean_target_det, log_std_target_det, mean_online, log_std_target_det</div>
  <div class="math-annotated-code-line">            )  # (B,D)</div>
  <div class="math-annotated-code-line">            kl_std = self._kl_diag_gaussian_per_dim(</div>
  <div class="math-annotated-code-line">                mean_target_det, log_std_target_det, mean_target_det, log_std_online</div>
  <div class="math-annotated-code-line">            )  # (B,D)</div>
  <div class="math-annotated-code-line">            mean_kl_mean = kl_mean.mean(dim=0)  # (D,)</div>
  <div class="math-annotated-code-line">            mean_kl_std = kl_std.mean(dim=0)  # (D,)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            loss_kl_mean = (alpha_mean_det * mean_kl_mean).sum()</div>
  <div class="math-annotated-code-line">            loss_kl_std = (alpha_std_det * mean_kl_std).sum()</div>
  <div class="math-annotated-code-line">            loss_kl_penalty = loss_kl_mean + loss_kl_std</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi}^{total} = \mathcal{L}_{\pi} + \mathcal{L}_{KL}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            policy_total_loss = loss_policy + loss_kl_penalty</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            self.policy_opt.zero_grad()</div>
  <div class="math-annotated-code-line">            policy_total_loss.backward()</div>
  <div class="math-annotated-code-line">            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)</div>
  <div class="math-annotated-code-line">            self.policy_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # snapshot after M-step</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            params_after = (</div>
  <div class="math-annotated-code-line">                nn.utils.parameters_to_vector(self.policy.parameters()).detach().clone()</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            param_delta = torch.norm(params_after - params_before)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Increment step counter for target-sync cadence bookkeeping.</div>
  <div class="math-annotated-code-line">        self._num_steps += 1</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Diagnostics for training monitoring.</div>
  <div class="math-annotated-code-line">        temperature_val = float(</div>
  <div class="math-annotated-code-line">            (F.softplus(self.log_temperature) + 1e-8).detach().item()</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{H}(q) = -\mathbb{E}_b\sum_i q_{b,i}\log q_{b,i}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        return {</div>
  <div class="math-annotated-code-line">            &quot;train/param_delta&quot;: float(param_delta.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/q&quot;: float(q_loss.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/policy&quot;: float(loss_policy.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/dual_eta&quot;: float(loss_temperature.detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/dual&quot;: float(dual_loss.detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/q_pi&quot;: float(kl_q_rel.detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/mean&quot;: float(mean_kl_mean.mean().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/std&quot;: float(mean_kl_std.mean().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;eta&quot;: temperature_val,</div>
  <div class="math-annotated-code-line">            &quot;lambda&quot;: float(</div>
  <div class="math-annotated-code-line">                (F.softplus(self.log_alpha_mean) + 1e-8).mean().detach().item()</div>
  <div class="math-annotated-code-line">            ),</div>
  <div class="math-annotated-code-line">            &quot;alpha_mean&quot;: float(</div>
  <div class="math-annotated-code-line">                (F.softplus(self.log_alpha_mean) + 1e-8).mean().detach().item()</div>
  <div class="math-annotated-code-line">            ),</div>
  <div class="math-annotated-code-line">            &quot;alpha_std&quot;: float(</div>
  <div class="math-annotated-code-line">                (F.softplus(self.log_alpha_stddev) + 1e-8).mean().detach().item()</div>
  <div class="math-annotated-code-line">            ),</div>
  <div class="math-annotated-code-line">            &quot;q/min&quot;: float(q_vals.min().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;q/max&quot;: float(q_vals.max().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;pi/std_min&quot;: float(std_online.min().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;pi/std_max&quot;: float(std_online.max().detach().item()),</div>
  <div class="math-annotated-code-line">            &quot;entropy&quot;: float(entropy.detach().item()),</div>
  <div class="math-annotated-code-line">        }</div>
</div>
