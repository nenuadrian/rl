# `trainers.vmpo.agent` Math-Annotated Source

_Source: `minerva/trainers/vmpo/agent.py`_

The full file is rendered in one continuous code-style block, with each `# LaTeX:` marker replaced inline by a rendered formula.

## Annotated Source

<div class="math-annotated-codeblock">
  <div class="math-annotated-code-line">from __future__ import annotations</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import math</div>
  <div class="math-annotated-code-line">from typing import Tuple, Dict, Any, Literal</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">import numpy as np</div>
  <div class="math-annotated-code-line">import torch</div>
  <div class="math-annotated-code-line">import torch.nn as nn</div>
  <div class="math-annotated-code-line">import torch.nn.functional as F</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">from minerva.trainers.vmpo.gaussian_mlp_policy import SquashedGaussianPolicy</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">class VMPOAgent:</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def __init__(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs_dim: int,</div>
  <div class="math-annotated-code-line">        act_dim: int,</div>
  <div class="math-annotated-code-line">        action_low: np.ndarray,</div>
  <div class="math-annotated-code-line">        action_high: np.ndarray,</div>
  <div class="math-annotated-code-line">        device: torch.device,</div>
  <div class="math-annotated-code-line">        policy_layer_sizes: Tuple[int, ...] = (256, 256),</div>
  <div class="math-annotated-code-line">        value_layer_sizes: Tuple[int, ...] = (256, 256),</div>
  <div class="math-annotated-code-line">        normalize_advantages: bool = True,</div>
  <div class="math-annotated-code-line">        gamma: float = 0.99,</div>
  <div class="math-annotated-code-line">        advantage_estimator: Literal[&quot;returns&quot;, &quot;dae&quot;, &quot;gae&quot;] = &quot;returns&quot;,</div>
  <div class="math-annotated-code-line">        gae_lambda: float = 0.95,</div>
  <div class="math-annotated-code-line">        policy_lr: float = 5e-4,</div>
  <div class="math-annotated-code-line">        value_lr: float = 1e-3,</div>
  <div class="math-annotated-code-line">        topk_fraction: float = 0.5,</div>
  <div class="math-annotated-code-line">        temperature_init: float = 1.0,</div>
  <div class="math-annotated-code-line">        temperature_lr: float = 1e-4,</div>
  <div class="math-annotated-code-line">        epsilon_eta: float = 0.1,</div>
  <div class="math-annotated-code-line">        epsilon_mu: float = 0.01,</div>
  <div class="math-annotated-code-line">        epsilon_sigma: float = 0.01,</div>
  <div class="math-annotated-code-line">        alpha_lr: float = 1e-4,</div>
  <div class="math-annotated-code-line">        max_grad_norm: float = 10.0,</div>
  <div class="math-annotated-code-line">        optimizer_type: str = &quot;adam&quot;,</div>
  <div class="math-annotated-code-line">        sgd_momentum: float = 0.9,</div>
  <div class="math-annotated-code-line">        shared_encoder: bool = False,</div>
  <div class="math-annotated-code-line">        ppo_like_backbone: bool = False,</div>
  <div class="math-annotated-code-line">        m_steps: int = 1,</div>
  <div class="math-annotated-code-line">    ):</div>
  <div class="math-annotated-code-line">        self.device = device</div>
  <div class="math-annotated-code-line">        self.normalize_advantages = bool(normalize_advantages)</div>
  <div class="math-annotated-code-line">        self.gamma = float(gamma)</div>
  <div class="math-annotated-code-line">        self.advantage_estimator = advantage_estimator</div>
  <div class="math-annotated-code-line">        self.gae_lambda = float(gae_lambda)</div>
  <div class="math-annotated-code-line">        self.policy_lr = float(policy_lr)</div>
  <div class="math-annotated-code-line">        self.value_lr = float(value_lr)</div>
  <div class="math-annotated-code-line">        self.topk_fraction = float(topk_fraction)</div>
  <div class="math-annotated-code-line">        self.temperature_init = float(temperature_init)</div>
  <div class="math-annotated-code-line">        self.temperature_lr = float(temperature_lr)</div>
  <div class="math-annotated-code-line">        self.epsilon_eta = float(epsilon_eta)</div>
  <div class="math-annotated-code-line">        self.epsilon_mu = float(epsilon_mu)</div>
  <div class="math-annotated-code-line">        self.epsilon_sigma = float(epsilon_sigma)</div>
  <div class="math-annotated-code-line">        self.alpha_lr = float(alpha_lr)</div>
  <div class="math-annotated-code-line">        self.max_grad_norm = float(max_grad_norm)</div>
  <div class="math-annotated-code-line">        self.optimizer_type = str(optimizer_type)</div>
  <div class="math-annotated-code-line">        self.sgd_momentum = float(sgd_momentum)</div>
  <div class="math-annotated-code-line">        self.m_steps = int(m_steps)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.policy = SquashedGaussianPolicy(</div>
  <div class="math-annotated-code-line">            obs_dim=obs_dim,</div>
  <div class="math-annotated-code-line">            act_dim=act_dim,</div>
  <div class="math-annotated-code-line">            policy_layer_sizes=policy_layer_sizes,</div>
  <div class="math-annotated-code-line">            value_layer_sizes=value_layer_sizes,</div>
  <div class="math-annotated-code-line">            action_low=action_low,</div>
  <div class="math-annotated-code-line">            action_high=action_high,</div>
  <div class="math-annotated-code-line">            shared_encoder=shared_encoder,</div>
  <div class="math-annotated-code-line">            ppo_like_backbone=ppo_like_backbone,</div>
  <div class="math-annotated-code-line">        ).to(device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        value_params = list(self.policy.value_head.parameters())</div>
  <div class="math-annotated-code-line">        if not self.policy.shared_encoder:</div>
  <div class="math-annotated-code-line">            value_params = list(self.policy.value_encoder.parameters()) + value_params</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tilde{\lambda}_{\pi} = \frac{\lambda_{\pi}}{\sqrt{M}}
\]</div>
  </div>
  <div class="math-annotated-code-line">        policy_lr_eff = self.policy_lr</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tilde{\lambda}_{V} = \frac{\lambda_{V}}{\sqrt{M}}
\]</div>
  </div>
  <div class="math-annotated-code-line">        value_lr_eff = self.value_lr</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.combined_opt = self._build_optimizer(</div>
  <div class="math-annotated-code-line">            [</div>
  <div class="math-annotated-code-line">                {</div>
  <div class="math-annotated-code-line">                    &quot;params&quot;: self.policy.policy_encoder.parameters(),</div>
  <div class="math-annotated-code-line">                    &quot;lr&quot;: policy_lr_eff,</div>
  <div class="math-annotated-code-line">                },</div>
  <div class="math-annotated-code-line">                {&quot;params&quot;: self.policy.policy_mean.parameters(), &quot;lr&quot;: policy_lr_eff},</div>
  <div class="math-annotated-code-line">                {&quot;params&quot;: self.policy.policy_logstd_parameters(), &quot;lr&quot;: policy_lr_eff},</div>
  <div class="math-annotated-code-line">                {&quot;params&quot;: value_params, &quot;lr&quot;: value_lr_eff},</div>
  <div class="math-annotated-code-line">            ]</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.opt = self.combined_opt</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Lagrange Multipliers (Dual Variables)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Temperature (eta) for advantage weighting</div>
  <div class="math-annotated-code-line">        temperature_init_t = torch.tensor(</div>
  <div class="math-annotated-code-line">            self.temperature_init, dtype=torch.float32, device=device</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)</div>
  <div class="math-annotated-code-line">        self.log_temperature = nn.Parameter(torch.log(torch.expm1(temperature_init_t)))</div>
  <div class="math-annotated-code-line">        self.eta_opt = self._build_optimizer(</div>
  <div class="math-annotated-code-line">            [self.log_temperature], lr=self.temperature_lr</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # KL Penalties (alpha) for trust region</div>
  <div class="math-annotated-code-line">        def inv_softplus(x):</div>
  <div class="math-annotated-code-line">            return np.log(np.expm1(x))</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.log_alpha_mu = nn.Parameter(</div>
  <div class="math-annotated-code-line">            torch.tensor(inv_softplus(1.0), dtype=torch.float32, device=device)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line">        self.log_alpha_sigma = nn.Parameter(</div>
  <div class="math-annotated-code-line">            torch.tensor(inv_softplus(1.0), dtype=torch.float32, device=device)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tilde{\lambda}_{\alpha} = \frac{\lambda_{\alpha}}{\sqrt{M}}
\]</div>
  </div>
  <div class="math-annotated-code-line">        effective_alpha_lr = self.alpha_lr #/ math.sqrt(self.m_steps)</div>
  <div class="math-annotated-code-line">        self.alpha_opt = self._build_optimizer(</div>
  <div class="math-annotated-code-line">            [self.log_alpha_mu, self.log_alpha_sigma], lr=effective_alpha_lr</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def _build_optimizer(</div>
  <div class="math-annotated-code-line">        self, params, lr: float | None = None</div>
  <div class="math-annotated-code-line">    ) -&gt; torch.optim.Optimizer:</div>
  <div class="math-annotated-code-line">        optimizer_type = self.optimizer_type.strip().lower()</div>
  <div class="math-annotated-code-line">        kwargs: dict[str, float] = {}</div>
  <div class="math-annotated-code-line">        if lr is not None:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;lr&quot;] = float(lr)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if optimizer_type == &quot;adam&quot;:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;eps&quot;] = 1e-5</div>
  <div class="math-annotated-code-line">            return torch.optim.Adam(params, **kwargs)</div>
  <div class="math-annotated-code-line">        if optimizer_type == &quot;sgd&quot;:</div>
  <div class="math-annotated-code-line">            kwargs[&quot;momentum&quot;] = float(self.sgd_momentum)</div>
  <div class="math-annotated-code-line">            return torch.optim.SGD(params, **kwargs)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def act(</div>
  <div class="math-annotated-code-line">        self,</div>
  <div class="math-annotated-code-line">        obs: np.ndarray,</div>
  <div class="math-annotated-code-line">        deterministic: bool = False,</div>
  <div class="math-annotated-code-line">    ) -&gt; Tuple[np.ndarray, float, np.ndarray, np.ndarray]:</div>
  <div class="math-annotated-code-line">        obs_np = np.asarray(obs)</div>
  <div class="math-annotated-code-line">        is_batch = obs_np.ndim &gt; 1</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if not is_batch:</div>
  <div class="math-annotated-code-line">            obs_np = obs_np[None, ...]  # Add batch dim</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            mean, log_std, value = self.policy.forward_all(obs_t)</div>
  <div class="math-annotated-code-line">            action_t, _ = self.policy.sample_action(mean, log_std, deterministic)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        action_np = action_t.cpu().numpy()</div>
  <div class="math-annotated-code-line">        value_np = value.cpu().numpy().squeeze(-1)</div>
  <div class="math-annotated-code-line">        mean_np = mean.cpu().numpy()</div>
  <div class="math-annotated-code-line">        log_std_np = log_std.cpu().numpy()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        if not is_batch:</div>
  <div class="math-annotated-code-line">            return (</div>
  <div class="math-annotated-code-line">                action_np[0],</div>
  <div class="math-annotated-code-line">                float(value_np.item()),</div>
  <div class="math-annotated-code-line">                mean_np[0],</div>
  <div class="math-annotated-code-line">                log_std_np[0],</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">        return action_np, value_np, mean_np, log_std_np</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def value(self, obs: np.ndarray) -&gt; float | np.ndarray:</div>
  <div class="math-annotated-code-line">        obs_np = np.asarray(obs)</div>
  <div class="math-annotated-code-line">        is_batch = obs_np.ndim &gt; 1</div>
  <div class="math-annotated-code-line">        if not is_batch:</div>
  <div class="math-annotated-code-line">            obs_np = obs_np[None, ...]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            v = self.policy.get_value(obs_t).squeeze(-1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        v_np = v.cpu().numpy()</div>
  <div class="math-annotated-code-line">        if not is_batch:</div>
  <div class="math-annotated-code-line">            return float(v_np.item())</div>
  <div class="math-annotated-code-line">        return v_np</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">    def update(self, batch: Dict[str, Any]) -&gt; Dict[str, float]:</div>
  <div class="math-annotated-code-line">        obs = batch[&quot;obs&quot;]</div>
  <div class="math-annotated-code-line">        actions = batch[&quot;actions&quot;]</div>
  <div class="math-annotated-code-line">        old_means = batch[&quot;old_means&quot;]</div>
  <div class="math-annotated-code-line">        old_log_stds = batch[&quot;old_log_stds&quot;]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        returns_raw = batch[&quot;returns&quot;].squeeze(-1)</div>
  <div class="math-annotated-code-line">        advantages = batch[&quot;advantages&quot;].squeeze(-1)</div>
  <div class="math-annotated-code-line">        restarting_weights = batch.get(&quot;restarting_weights&quot;, None)</div>
  <div class="math-annotated-code-line">        importance_weights = batch.get(&quot;importance_weights&quot;, None)</div>
  <div class="math-annotated-code-line">        if restarting_weights is None:</div>
  <div class="math-annotated-code-line">            restarting_weights = torch.ones_like(advantages)</div>
  <div class="math-annotated-code-line">        else:</div>
  <div class="math-annotated-code-line">            restarting_weights = restarting_weights.squeeze(-1)</div>
  <div class="math-annotated-code-line">        if importance_weights is None:</div>
  <div class="math-annotated-code-line">            importance_weights = torch.ones_like(advantages)</div>
  <div class="math-annotated-code-line">        else:</div>
  <div class="math-annotated-code-line">            importance_weights = importance_weights.squeeze(-1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Advantage normalization</div>
  <div class="math-annotated-code-line">        if self.normalize_advantages:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}
\]</div>
  </div>
  <div class="math-annotated-code-line">            advantages = (advantages - advantages.mean()) / (</div>
  <div class="math-annotated-code-line">                advantages.std(unbiased=False) + 1e-8</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Capture params before update for &#x27;param_delta&#x27;</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            params_before = nn.utils.parameters_to_vector(</div>
  <div class="math-annotated-code-line">                self.policy.parameters()</div>
  <div class="math-annotated-code-line">            ).detach()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line">        # E-Step: Re-weighting advantages (DeepMind-style semantics)</div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\eta = \operatorname{softplus}(\tilde{\eta}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">        eta = F.softplus(self.log_temperature) + 1e-8</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
s_t = \frac{w_t^{restart}\hat{A}_t}{\eta}
\]</div>
  </div>
  <div class="math-annotated-code-line">        scaled_advantages = (restarting_weights * advantages) / eta</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Numerical stability: subtract global max before exponentiation.</div>
  <div class="math-annotated-code-line">        max_scaled_advantage = scaled_advantages.max().detach()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            if not 0.0 &lt; self.topk_fraction &lt;= 1.0:</div>
  <div class="math-annotated-code-line">                raise ValueError(</div>
  <div class="math-annotated-code-line">                    f&quot;`topk_fraction` must be in (0, 1], got {self.topk_fraction}&quot;</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            if self.topk_fraction &lt; 1.0:</div>
  <div class="math-annotated-code-line">                # Exclude restarting states from top-k selection.</div>
  <div class="math-annotated-code-line">                valid_scaled_advantages = scaled_advantages.detach().clone()</div>
  <div class="math-annotated-code-line">                valid_scaled_advantages[restarting_weights &lt;= 0.0] = -torch.inf</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
k = \lfloor \rho N \rfloor
\]</div>
  </div>
  <div class="math-annotated-code-line">                k = int(self.topk_fraction * valid_scaled_advantages.numel())</div>
  <div class="math-annotated-code-line">                if k &lt;= 0:</div>
  <div class="math-annotated-code-line">                    raise ValueError(</div>
  <div class="math-annotated-code-line">                        &quot;topk_fraction too low to select any scaled advantages.&quot;</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                topk_vals, _ = torch.topk(valid_scaled_advantages, k)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tau = \min(\operatorname{TopK}(s, k))
\]</div>
  </div>
  <div class="math-annotated-code-line">                threshold = topk_vals.min()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
m_t = \mathbf{1}[s_t \ge \tau]
\]</div>
  </div>
  <div class="math-annotated-code-line">                topk_weights = (valid_scaled_advantages &gt;= threshold).to(</div>
  <div class="math-annotated-code-line">                    restarting_weights.dtype</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\bar{w}_t = w_t^{restart} \cdot m_t
\]</div>
  </div>
  <div class="math-annotated-code-line">                topk_restarting_weights = restarting_weights * topk_weights</div>
  <div class="math-annotated-code-line">            else:</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tau = \min_t s_t
\]</div>
  </div>
  <div class="math-annotated-code-line">                threshold = scaled_advantages.detach().min()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\bar{w}_t = w_t^{restart}
\]</div>
  </div>
  <div class="math-annotated-code-line">                topk_restarting_weights = restarting_weights</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # Mask for selected samples (used for KL terms and diagnostics).</div>
  <div class="math-annotated-code-line">            mask_bool = topk_restarting_weights &gt; 0.0</div>
  <div class="math-annotated-code-line">            if not bool(mask_bool.any()):</div>
  <div class="math-annotated-code-line">                # Fallback to avoid empty reductions on tiny rollouts.</div>
  <div class="math-annotated-code-line">                mask_bool = torch.ones_like(mask_bool, dtype=torch.bool)</div>
  <div class="math-annotated-code-line">                topk_restarting_weights = torch.ones_like(topk_restarting_weights)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Use stop-gradient semantics for importance weights.</div>
  <div class="math-annotated-code-line">        importance_weights_sg = importance_weights.detach()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\tilde{\psi}_t = \bar{w}_t \, w_t^{imp} \exp(s_t - s_{\max})
\]</div>
  </div>
  <div class="math-annotated-code-line">        unnormalized_weights = (</div>
  <div class="math-annotated-code-line">            topk_restarting_weights</div>
  <div class="math-annotated-code-line">            * importance_weights_sg</div>
  <div class="math-annotated-code-line">            * torch.exp(scaled_advantages - max_scaled_advantage)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
Z = \sum_t \tilde{\psi}_t
\]</div>
  </div>
  <div class="math-annotated-code-line">        sum_weights = unnormalized_weights.sum() + 1e-8</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
N_{sel} = \sum_t \bar{w}_t
\]</div>
  </div>
  <div class="math-annotated-code-line">        num_samples = topk_restarting_weights.sum() + 1e-8</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\psi_t = \frac{\tilde{\psi}_t}{Z}
\]</div>
  </div>
  <div class="math-annotated-code-line">        weights = unnormalized_weights / sum_weights</div>
  <div class="math-annotated-code-line">        weights_detached = weights.detach()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\log \bar{\psi} = \log Z + s_{\max} - \log N_{sel}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        log_mean_weights = (</div>
  <div class="math-annotated-code-line">            torch.log(sum_weights) + max_scaled_advantage - torch.log(num_samples)</div>
  <div class="math-annotated-code-line">        )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\eta} = \eta \left(\epsilon_{\eta} + \log \bar{\psi}\right)
\]</div>
  </div>
  <div class="math-annotated-code-line">        dual_loss = eta * (self.epsilon_eta + log_mean_weights)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        self.eta_opt.zero_grad()</div>
  <div class="math-annotated-code-line">        dual_loss.backward()</div>
  <div class="math-annotated-code-line">        self.eta_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # Compute Final Weights for Policy</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\eta&#x27; = \operatorname{softplus}(\tilde{\eta}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">            eta_final = F.softplus(self.log_temperature) + 1e-8</div>
  <div class="math-annotated-code-line">            # Logging: effective sample size and selection stats.</div>
  <div class="math-annotated-code-line">            ess = 1.0 / (weights_detached.pow(2).sum() + 1e-12)</div>
  <div class="math-annotated-code-line">            selected_frac = float(mask_bool.float().mean().item())</div>
  <div class="math-annotated-code-line">            adv_std_over_temperature = (</div>
  <div class="math-annotated-code-line">                advantages.std(unbiased=False) / (eta_final + 1e-12)</div>
  <div class="math-annotated-code-line">            ).item()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line">        # M-Step: Policy &amp; Value Update</div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        for _ in range(self.m_steps):  # Multiple epochs of optimization per batch</div>
  <div class="math-annotated-code-line">            # Run forward pass on ALL observations</div>
  <div class="math-annotated-code-line">            current_mean, current_log_std, v_pred_fw = self.policy.forward_all(obs)</div>
  <div class="math-annotated-code-line">            v_pred = v_pred_fw.squeeze(-1)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- Policy Loss --</div>
  <div class="math-annotated-code-line">            log_prob = self.policy.log_prob(</div>
  <div class="math-annotated-code-line">                current_mean, current_log_std, actions</div>
  <div class="math-annotated-code-line">            ).squeeze(-1)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi}^{NLL} = -\sum_t \psi_t \log \pi_{\theta}(a_t|s_t)
\]</div>
  </div>
  <div class="math-annotated-code-line">            weighted_nll = -(weights_detached * log_prob).sum()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- KL Divergence (Full Batch Diagnostics) --</div>
  <div class="math-annotated-code-line">            # We compute this for logging purposes to see global drift</div>
  <div class="math-annotated-code-line">            with torch.no_grad():</div>
  <div class="math-annotated-code-line">                old_std = old_log_stds.exp()</div>
  <div class="math-annotated-code-line">                new_std = current_log_std.exp()</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
D_{\mu}^{all} = \mathbb{E}\left[\sum_j \frac{(\mu_j-\mu_j^{old})^2}{2(\sigma_j^{old})^2}\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">                kl_mean_all = (</div>
  <div class="math-annotated-code-line">                    ((current_mean - old_means) ** 2 / (2.0 * old_std**2 + 1e-8))</div>
  <div class="math-annotated-code-line">                    .sum(dim=-1)</div>
  <div class="math-annotated-code-line">                    .mean()</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
D_{\sigma}^{all} = \mathbb{E}\left[\frac{1}{2}\sum_j\left(\frac{(\sigma_j^{old})^2}{\sigma_j^2} - 1 + 2(\log \sigma_j - \log \sigma_j^{old})\right)\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">                kl_std_all = (</div>
  <div class="math-annotated-code-line">                    0.5</div>
  <div class="math-annotated-code-line">                    * (</div>
  <div class="math-annotated-code-line">                        (old_std**2) / (new_std**2 + 1e-8)</div>
  <div class="math-annotated-code-line">                        - 1.0</div>
  <div class="math-annotated-code-line">                        + 2.0 * (current_log_std - old_log_stds)</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                    .sum(dim=-1)</div>
  <div class="math-annotated-code-line">                    .mean()</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- KL Divergence (Selected Samples for Optimization) --</div>
  <div class="math-annotated-code-line">            mean_sel = current_mean[mask_bool]</div>
  <div class="math-annotated-code-line">            log_std_sel = current_log_std[mask_bool]</div>
  <div class="math-annotated-code-line">            old_mean_sel = old_means[mask_bool]</div>
  <div class="math-annotated-code-line">            old_log_std_sel = old_log_stds[mask_bool]</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            old_std_sel = old_log_std_sel.exp()</div>
  <div class="math-annotated-code-line">            new_std_sel = log_std_sel.exp()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # Decoupled KL</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
D_{\mu} = \mathbb{E}_{t \in \mathcal{S}}\left[\sum_j \frac{(\mu_j-\mu_j^{old})^2}{2(\sigma_j^{old})^2}\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">            kl_mu_sel = (</div>
  <div class="math-annotated-code-line">                (0.5 * ((mean_sel - old_mean_sel) ** 2 / (old_std_sel**2 + 1e-8)))</div>
  <div class="math-annotated-code-line">                .sum(dim=-1)</div>
  <div class="math-annotated-code-line">                .mean()</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
D_{\sigma} = \mathbb{E}_{t \in \mathcal{S}}\left[\frac{1}{2}\sum_j\left(\frac{(\sigma_j^{old})^2}{\sigma_j^2} - 1 + 2(\log \sigma_j - \log \sigma_j^{old})\right)\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">            kl_sigma_sel = (</div>
  <div class="math-annotated-code-line">                (</div>
  <div class="math-annotated-code-line">                    0.5</div>
  <div class="math-annotated-code-line">                    * (</div>
  <div class="math-annotated-code-line">                        (old_std_sel**2) / (new_std_sel**2 + 1e-8)</div>
  <div class="math-annotated-code-line">                        - 1.0</div>
  <div class="math-annotated-code-line">                        + 2.0 * (log_std_sel - old_log_std_sel)</div>
  <div class="math-annotated-code-line">                    )</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                .sum(dim=-1)</div>
  <div class="math-annotated-code-line">                .mean()</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- Alpha Optimization --</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\mu} = \operatorname{softplus}(\tilde{\alpha}_{\mu}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">            alpha_mu = F.softplus(self.log_alpha_mu) + 1e-8</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\alpha_{\sigma} = \operatorname{softplus}(\tilde{\alpha}_{\sigma}) + \epsilon
\]</div>
  </div>
  <div class="math-annotated-code-line">            alpha_sigma = F.softplus(self.log_alpha_sigma) + 1e-8</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # We minimize: alpha * (epsilon - KL)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\alpha} = \alpha_{\mu}(\epsilon_{\mu} - D_{\mu}) + \alpha_{\sigma}(\epsilon_{\sigma} - D_{\sigma})
\]</div>
  </div>
  <div class="math-annotated-code-line">            alpha_loss = alpha_mu * (</div>
  <div class="math-annotated-code-line">                self.epsilon_mu - kl_mu_sel.detach()</div>
  <div class="math-annotated-code-line">            ) + alpha_sigma * (self.epsilon_sigma - kl_sigma_sel.detach())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            self.alpha_opt.zero_grad()</div>
  <div class="math-annotated-code-line">            alpha_loss.backward()</div>
  <div class="math-annotated-code-line">            self.alpha_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- Final Policy Loss --</div>
  <div class="math-annotated-code-line">            with torch.no_grad():</div>
  <div class="math-annotated-code-line">                alpha_mu_det = F.softplus(self.log_alpha_mu).detach() + 1e-8</div>
  <div class="math-annotated-code-line">                alpha_sigma_det = F.softplus(self.log_alpha_sigma).detach() + 1e-8</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{\pi} = \mathcal{L}_{\pi}^{NLL} + \alpha_{\mu}D_{\mu} + \alpha_{\sigma}D_{\sigma}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            policy_loss = (</div>
  <div class="math-annotated-code-line">                weighted_nll</div>
  <div class="math-annotated-code-line">                + (alpha_mu_det * kl_mu_sel)</div>
  <div class="math-annotated-code-line">                + (alpha_sigma_det * kl_sigma_sel)</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # -- Critic Loss --</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L}_{V} = \frac{1}{2}\mathbb{E}\left[(V_{\phi}(s_t) - R_t)^2\right]
\]</div>
  </div>
  <div class="math-annotated-code-line">            value_loss = 0.5 * F.mse_loss(v_pred, returns_raw.detach())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{L} = \mathcal{L}_{\pi} + \mathcal{L}_{V}
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            total_loss = policy_loss + value_loss</div>
  <div class="math-annotated-code-line">            self.combined_opt.zero_grad()</div>
  <div class="math-annotated-code-line">            total_loss.backward()</div>
  <div class="math-annotated-code-line">            nn.utils.clip_grad_norm_(</div>
  <div class="math-annotated-code-line">                list(self.policy.policy_encoder.parameters())</div>
  <div class="math-annotated-code-line">                + list(self.policy.policy_mean.parameters())</div>
  <div class="math-annotated-code-line">                + self.policy.policy_logstd_parameters()</div>
  <div class="math-annotated-code-line">                + (</div>
  <div class="math-annotated-code-line">                    []</div>
  <div class="math-annotated-code-line">                    if self.policy.shared_encoder</div>
  <div class="math-annotated-code-line">                    else list(self.policy.value_encoder.parameters())</div>
  <div class="math-annotated-code-line">                )</div>
  <div class="math-annotated-code-line">                + list(self.policy.value_head.parameters()),</div>
  <div class="math-annotated-code-line">                self.max_grad_norm,</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            self.combined_opt.step()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line">        # Post-Update Diagnostics</div>
  <div class="math-annotated-code-line">        # ================================================================</div>
  <div class="math-annotated-code-line">        with torch.no_grad():</div>
  <div class="math-annotated-code-line">            # 1. Parameter Delta</div>
  <div class="math-annotated-code-line">            params_after = nn.utils.parameters_to_vector(</div>
  <div class="math-annotated-code-line">                self.policy.parameters()</div>
  <div class="math-annotated-code-line">            ).detach()</div>
  <div class="math-annotated-code-line">            param_delta = torch.norm(params_after - params_before).item()</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # 2. Explained Variance (Unnormalized)</div>
  <div class="math-annotated-code-line">            v_pred = self.policy.get_value(obs).squeeze(-1)</div>
  <div class="math-annotated-code-line">            y = returns_raw</div>
  <div class="math-annotated-code-line">            var_y = y.var(unbiased=False)</div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\operatorname{EV} = 1 - \frac{\operatorname{Var}[R - V]}{\operatorname{Var}[R] + \epsilon}
\]</div>
  </div>
  <div class="math-annotated-code-line">            explained_var = 1.0 - (y - v_pred).var(unbiased=False) / (var_y + 1e-8)</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            # 3. Action Saturation</div>
  <div class="math-annotated-code-line">            mean_eval, log_std_eval = self.policy(obs)</div>
  <div class="math-annotated-code-line">            action_eval, _ = self.policy.sample_action(</div>
  <div class="math-annotated-code-line">                mean_eval, log_std_eval, deterministic=True</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line">            mean_abs_action = float(action_eval.abs().mean().item())</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line math-annotated-code-formula">
    <div class="arithmatex">\[
\mathcal{H} = \mathbb{E}\left[\sum_j \frac{1}{2}\left(1 + \log(2\pi\sigma_j^2)\right)\right]
\]</div>
  </div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">            entropy = (</div>
  <div class="math-annotated-code-line">                (0.5 * (1 + torch.log(2 * torch.pi * new_std_sel**2))).sum(-1).mean()</div>
  <div class="math-annotated-code-line">            )</div>
  <div class="math-annotated-code-line"> </div>
  <div class="math-annotated-code-line">        return {</div>
  <div class="math-annotated-code-line">            &quot;loss/total&quot;: float(total_loss.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/value&quot;: float(value_loss.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/policy&quot;: float(policy_loss.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/policy_weighted_nll&quot;: float(weighted_nll.item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/policy_kl_mean_pen&quot;: float((alpha_mu_det * kl_mu_sel).item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/policy_kl_std_pen&quot;: float((alpha_sigma_det * kl_sigma_sel).item()),</div>
  <div class="math-annotated-code-line">            &quot;loss/alpha&quot;: float(alpha_loss.item()),</div>
  <div class="math-annotated-code-line">            # KL Diagnostics</div>
  <div class="math-annotated-code-line">            &quot;kl/mean&quot;: float(kl_mean_all.item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/std&quot;: float(kl_std_all.item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/mean_sel&quot;: float(kl_mu_sel.item()),</div>
  <div class="math-annotated-code-line">            &quot;kl/std_sel&quot;: float(kl_sigma_sel.item()),</div>
  <div class="math-annotated-code-line">            # Dual Variables</div>
  <div class="math-annotated-code-line">            &quot;vmpo/alpha_mu&quot;: float(alpha_mu_det.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/alpha_sigma&quot;: float(alpha_sigma_det.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/dual_loss&quot;: float(dual_loss.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/epsilon_eta&quot;: float(self.epsilon_eta),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/temperature_raw&quot;: float(eta_final.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/adv_std_over_temperature&quot;: float(adv_std_over_temperature),</div>
  <div class="math-annotated-code-line">            # Selection Stats</div>
  <div class="math-annotated-code-line">            &quot;vmpo/selected_frac&quot;: float(selected_frac),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/threshold&quot;: float(threshold.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/ess&quot;: float(ess.item()),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/restarting_frac&quot;: float(</div>
  <div class="math-annotated-code-line">                (restarting_weights &lt;= 0.0).float().mean().item()</div>
  <div class="math-annotated-code-line">            ),</div>
  <div class="math-annotated-code-line">            &quot;vmpo/importance_mean&quot;: float(importance_weights_sg.mean().item()),</div>
  <div class="math-annotated-code-line">            # Training Dynamics</div>
  <div class="math-annotated-code-line">            &quot;train/entropy&quot;: float(entropy.item()),</div>
  <div class="math-annotated-code-line">            &quot;train/param_delta&quot;: float(param_delta),</div>
  <div class="math-annotated-code-line">            &quot;train/mean_abs_action&quot;: float(mean_abs_action),</div>
  <div class="math-annotated-code-line">            # Data Stats</div>
  <div class="math-annotated-code-line">            &quot;adv/raw_mean&quot;: float(advantages.mean().item()),</div>
  <div class="math-annotated-code-line">            &quot;adv/raw_std&quot;: float((advantages.std(unbiased=False) + 1e-8).item()),</div>
  <div class="math-annotated-code-line">            &quot;returns/raw_mean&quot;: float(returns_raw.mean().item()),</div>
  <div class="math-annotated-code-line">            &quot;returns/raw_std&quot;: float((returns_raw.std(unbiased=False) + 1e-8).item()),</div>
  <div class="math-annotated-code-line">            &quot;value/explained_var&quot;: float(explained_var.item()),</div>
  <div class="math-annotated-code-line">            &quot;value/pred_mean&quot;: float(v_pred.mean().item()),</div>
  <div class="math-annotated-code-line">            &quot;value/pred_std&quot;: float(v_pred.std(unbiased=False).item()),</div>
  <div class="math-annotated-code-line">        }</div>
</div>
