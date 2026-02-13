from __future__ import annotations

import os
import time
from collections import deque
from typing import Dict, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from trainers.ppo_trxl.pom_env import POM_ENV_ID, register_pom_env
from utils.env import wrap_record_video
from utils.wandb_utils import log_wandb


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(
        f"{key}={float(value):.4f}" for key, value in sorted(metrics.items())
    )


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
    return layer


def batched_index_select(input_tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    for axis in range(1, len(input_tensor.shape)):
        if axis != dim:
            index = index.unsqueeze(axis)
    expanse = list(input_tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_tensor, dim, index)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, min_timescale: float = 2.0, max_timescale: float = 1e4):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"TRxL dim must be even for sinusoidal encoding, got {dim}")
        freqs = torch.arange(0, dim, 2.0)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len: int) -> torch.Tensor:
        seq = torch.arange(seq_len - 1, -1, -1.0, device=self.inv_freqs.device)
        sinusoidal_inp = seq.unsqueeze(1) * self.inv_freqs.unsqueeze(0)
        return torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        if self.head_size * num_heads != embed_dim:
            raise ValueError(
                "Embedding dimension needs to be divisible by number of heads: "
                f"embed_dim={embed_dim}, num_heads={num_heads}"
            )

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(batch_size, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(batch_size, key_len, self.num_heads, self.head_size)
        query = query.reshape(batch_size, query_len, self.num_heads, self.head_size)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            batch_size,
            query_len,
            self.num_heads * self.head_size,
        )
        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value
        attention, attention_weights = self.attention(value, key, query_, mask)
        x = attention + query
        x_ = self.layer_norm_attn(x)
        forward = self.fc_projection(x_)
        out = forward + x
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        max_episode_steps: int,
        positional_encoding: str,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding.lower()
        if self.positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memories: torch.Tensor,
        mask: torch.Tensor,
        memory_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        out_memories: list[torch.Tensor] = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, _ = layer(memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask)
            x = x.squeeze(1)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

        return x, torch.stack(out_memories, dim=1)


class TrXLAgent(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space_shape: Sequence[int],
        *,
        trxl_num_layers: int,
        trxl_num_heads: int,
        trxl_dim: int,
        trxl_positional_encoding: str,
        max_episode_steps: int,
        reconstruction_coef: float,
    ):
        super().__init__()

        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(
                f"TRxL trainer expects Box observations, got {type(observation_space).__name__}"
            )

        if observation_space.shape is None:
            raise ValueError("Observation space has no shape.")

        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps
        self.trxl_dim = trxl_dim
        self.is_image_obs = len(self.obs_shape) > 1
        self.reconstruction_coef = float(reconstruction_coef)

        if self.is_image_obs:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, trxl_dim)),
                nn.ReLU(),
            )
        else:
            obs_dim = int(np.prod(np.array(self.obs_shape)))
            self.encoder = layer_init(nn.Linear(obs_dim, trxl_dim))

        self.transformer = Transformer(
            trxl_num_layers,
            trxl_dim,
            trxl_num_heads,
            max_episode_steps,
            trxl_positional_encoding,
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(trxl_dim, trxl_dim)),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(trxl_dim, out_features=int(num_actions)), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(trxl_dim, 1), 1)

        if self.reconstruction_coef > 0.0 and self.is_image_obs:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )
        else:
            self.transposed_cnn = None

        self._last_hidden: torch.Tensor | None = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_image_obs:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
            return x
        x = x.reshape(x.shape[0], -1)
        return self.encoder(x)

    def get_value(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self._encode(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_indices: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._encode(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self._last_hidden = x

        logits = [branch(x) for branch in self.actor_branches]
        probs = [Categorical(logits=logit) for logit in logits]

        if action is None:
            if deterministic:
                action = torch.stack([torch.argmax(logit, dim=-1) for logit in logits], dim=1)
            else:
                action = torch.stack([dist.sample() for dist in probs], dim=1)

        log_probs = [dist.log_prob(action[:, i]) for i, dist in enumerate(probs)]
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        return (
            action,
            torch.stack(log_probs, dim=1),
            entropies,
            self.critic(x).flatten(),
            memory,
        )

    def reconstruct_observation(self) -> torch.Tensor:
        if self.transposed_cnn is None or self._last_hidden is None:
            raise RuntimeError("Reconstruction requested but decoder/hidden state is unavailable.")
        x = self.transposed_cnn(self._last_hidden)
        return x.permute((0, 2, 3, 1))


class PPOTRxLTrainer:
    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        device: torch.device,
        num_envs: int,
        num_steps: int,
        num_minibatches: int,
        update_epochs: int,
        init_lr: float,
        final_lr: float,
        anneal_steps: int,
        gamma: float,
        gae_lambda: float,
        norm_adv: bool,
        clip_coef: float,
        clip_vloss: bool,
        init_ent_coef: float,
        final_ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        target_kl: float | None,
        trxl_num_layers: int,
        trxl_num_heads: int,
        trxl_dim: int,
        trxl_memory_length: int,
        trxl_positional_encoding: str,
        reconstruction_coef: float,
        capture_video: bool = False,
        run_name: str | None = None,
    ):
        self.env_id = env_id
        self.seed = int(seed)
        self.device = device
        self.capture_video = bool(capture_video)

        safe_run_name = (
            run_name if run_name is not None else f"ppo_trxl-{env_id}-seed{seed}"
        ).replace("/", "-")
        self.video_dir = os.path.join("videos", safe_run_name)

        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.num_minibatches = int(num_minibatches)
        self.update_epochs = int(update_epochs)

        self.init_lr = float(init_lr)
        self.final_lr = float(final_lr)
        self.anneal_steps = int(anneal_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.norm_adv = bool(norm_adv)
        self.clip_coef = float(clip_coef)
        self.clip_vloss = bool(clip_vloss)
        self.init_ent_coef = float(init_ent_coef)
        self.final_ent_coef = float(final_ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = None if target_kl is None else float(target_kl)

        self.trxl_num_layers = int(trxl_num_layers)
        self.trxl_num_heads = int(trxl_num_heads)
        self.trxl_dim = int(trxl_dim)
        self.trxl_memory_length = int(trxl_memory_length)
        self.trxl_positional_encoding = str(trxl_positional_encoding)
        self.reconstruction_coef = float(reconstruction_coef)

        if self.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive.")

        env_fns = [
            (lambda i=i: self._make_env(seed=self.seed + i, env_index=i, record_video=self.capture_video))
            for i in range(self.num_envs)
        ]

        self.env = gym.vector.SyncVectorEnv(env_fns)

        obs_space = self.env.single_observation_space
        action_space = self.env.single_action_space

        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(
                "PPO-TRxL currently supports Box observations only; "
                f"got {type(obs_space).__name__}."
            )

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_space_shape = (int(action_space.n),)
            self._is_single_discrete = True
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_space_shape = tuple(int(x) for x in action_space.nvec)
            self._is_single_discrete = False
        else:
            raise ValueError(
                "PPO-TRxL currently supports Discrete and MultiDiscrete action spaces only; "
                f"got {type(action_space).__name__}."
            )

        self.obs_shape = obs_space.shape

        self.max_episode_steps = self._resolve_max_episode_steps(self.env.envs[0])
        self.trxl_memory_length = min(self.trxl_memory_length, self.max_episode_steps)

        self.agent = TrXLAgent(
            observation_space=obs_space,
            action_space_shape=self.action_space_shape,
            trxl_num_layers=self.trxl_num_layers,
            trxl_num_heads=self.trxl_num_heads,
            trxl_dim=self.trxl_dim,
            trxl_positional_encoding=self.trxl_positional_encoding,
            max_episode_steps=self.max_episode_steps,
            reconstruction_coef=self.reconstruction_coef,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=self.init_lr)
        self.bce_loss = nn.BCELoss()

        self.memory_mask, self.memory_indices = self._build_memory_helpers(
            memory_length=self.trxl_memory_length,
            max_episode_steps=self.max_episode_steps,
            device=self.device,
        )

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.recent_returns: deque[float] = deque(maxlen=100)
        self.recent_lengths: deque[float] = deque(maxlen=100)
        self.last_eval = 0
        self.last_checkpoint = 0

    def _resolve_max_episode_steps(self, env: gym.Env) -> int:
        max_episode_steps = None

        if getattr(env, "spec", None) is not None:
            max_episode_steps = getattr(env.spec, "max_episode_steps", None)

        if not max_episode_steps:
            unwrapped = env.unwrapped
            max_episode_steps = getattr(unwrapped, "max_episode_steps", None)

        if not max_episode_steps or int(max_episode_steps) <= 0:
            max_episode_steps = 1024

        return int(max_episode_steps)

    def _make_env(self, *, seed: int, env_index: int, record_video: bool) -> gym.Env:
        should_record = bool(record_video and env_index == 0)
        render_mode = "rgb_array" if should_record else None

        if self.env_id == POM_ENV_ID:
            register_pom_env()

        if "MiniGrid" in self.env_id:
            from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

            env = gym.make(
                self.env_id,
                agent_view_size=3,
                tile_size=28,
                render_mode=render_mode,
            )
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
            env = gym.wrappers.TimeLimit(env, 96)
        else:
            if self.env_id != POM_ENV_ID:
                # Import for side effects when memory-gym env IDs are used.
                try:
                    import memory_gym  # noqa: F401
                except Exception:
                    pass
            env = gym.make(self.env_id, render_mode=render_mode)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if should_record:
            env = wrap_record_video(env, self.video_dir)

        return env

    @staticmethod
    def _build_memory_helpers(
        *,
        memory_length: int,
        max_episode_steps: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_mask = torch.tril(
            torch.ones((memory_length, memory_length), dtype=torch.bool, device=device),
            diagonal=-1,
        )

        if memory_length > 1:
            repetitions = torch.repeat_interleave(
                torch.arange(0, memory_length, device=device).unsqueeze(0),
                memory_length - 1,
                dim=0,
            ).long()
        else:
            repetitions = torch.empty((0, memory_length), dtype=torch.long, device=device)

        windows = torch.stack(
            [
                torch.arange(i, i + memory_length, device=device)
                for i in range(max_episode_steps - memory_length + 1)
            ]
        ).long()
        memory_indices = torch.cat((repetitions, windows), dim=0)

        return memory_mask, memory_indices

    def train(
        self,
        *,
        total_steps: int,
        eval_interval: int,
        save_interval: int,
        out_dir: str,
    ):
        total_steps = int(total_steps)
        eval_interval = int(eval_interval)
        save_interval = int(save_interval)

        batch_size = self.num_steps * self.num_envs
        if batch_size % self.num_minibatches != 0:
            raise ValueError(
                "batch_size must be divisible by num_minibatches: "
                f"batch_size={batch_size}, num_minibatches={self.num_minibatches}"
            )
        minibatch_size = batch_size // self.num_minibatches
        num_iterations = max(1, total_steps // batch_size)

        print(
            "[PPO-TRXL] training started: "
            f"total_steps={total_steps}, "
            f"effective_steps={num_iterations * batch_size}, "
            f"num_envs={self.num_envs}, "
            f"num_steps={self.num_steps}, "
            f"batch_size={batch_size}, "
            f"num_minibatches={self.num_minibatches}, "
            f"minibatch_size={minibatch_size}, "
            f"update_epochs={self.update_epochs}"
        )

        rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs, len(self.action_space_shape)),
            dtype=torch.long,
            device=self.device,
        )
        dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.obs_shape,
            device=self.device,
        )
        log_probs = torch.zeros(
            (self.num_steps, self.num_envs, len(self.action_space_shape)),
            device=self.device,
        )
        values = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        stored_memory_masks = torch.zeros(
            (self.num_steps, self.num_envs, self.trxl_memory_length),
            dtype=torch.bool,
            device=self.device,
        )
        stored_memory_index = torch.zeros(
            (self.num_steps, self.num_envs),
            dtype=torch.long,
            device=self.device,
        )
        stored_memory_indices = torch.zeros(
            (self.num_steps, self.num_envs, self.trxl_memory_length),
            dtype=torch.long,
            device=self.device,
        )

        global_step = 0
        start_time = time.time()

        next_obs_np, _ = self.env.reset(seed=self.seed)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        env_current_episode_step = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        next_memory = torch.zeros(
            (
                self.num_envs,
                self.max_episode_steps,
                self.trxl_num_layers,
                self.trxl_dim,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for iteration in range(1, num_iterations + 1):
            do_anneal = self.anneal_steps > 0 and global_step < self.anneal_steps
            frac = 1.0 - (global_step / float(self.anneal_steps)) if do_anneal else 0.0
            lr = (self.init_lr - self.final_lr) * frac + self.final_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            ent_coef = (self.init_ent_coef - self.final_ent_coef) * frac + self.final_ent_coef

            stored_memories = [next_memory[e] for e in range(self.num_envs)]
            for e in range(self.num_envs):
                stored_memory_index[:, e] = e

            for step in range(self.num_steps):
                global_step += self.num_envs

                with torch.no_grad():
                    obs[step] = next_obs
                    dones[step] = next_done
                    clipped_steps = torch.clamp(
                        env_current_episode_step,
                        0,
                        self.trxl_memory_length - 1,
                    )
                    stored_memory_masks[step] = self.memory_mask[clipped_steps]
                    stored_memory_indices[step] = self.memory_indices[env_current_episode_step]

                    memory_window = batched_index_select(
                        next_memory,
                        1,
                        stored_memory_indices[step],
                    )
                    action, logprob, _, value, new_memory = self.agent.get_action_and_value(
                        next_obs,
                        memory_window,
                        stored_memory_masks[step],
                        stored_memory_indices[step],
                    )
                    next_memory[
                        torch.arange(self.num_envs, device=self.device),
                        env_current_episode_step,
                    ] = new_memory
                    actions[step], log_probs[step], values[step] = action, logprob, value

                action_np = action.detach().cpu().numpy()
                if self._is_single_discrete:
                    action_exec = action_np.squeeze(-1)
                else:
                    action_exec = action_np

                next_obs_np, reward, terminations, truncations, _ = self.env.step(action_exec)
                done_np = np.logical_or(terminations, truncations)

                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device)

                for env_i, done_flag in enumerate(done_np):
                    self.episode_returns[env_i] += float(reward[env_i])
                    self.episode_lengths[env_i] += 1

                    if done_flag:
                        self.recent_returns.append(float(self.episode_returns[env_i]))
                        self.recent_lengths.append(float(self.episode_lengths[env_i]))
                        self.episode_returns[env_i] = 0.0
                        self.episode_lengths[env_i] = 0

                        env_current_episode_step[env_i] = 0
                        mem_index = int(stored_memory_index[step, env_i].item())
                        stored_memories[mem_index] = stored_memories[mem_index].clone()
                        next_memory[env_i].zero_()

                        if step < self.num_steps - 1:
                            stored_memories.append(next_memory[env_i])
                            stored_memory_index[step + 1 :, env_i] = len(stored_memories) - 1
                    else:
                        env_current_episode_step[env_i] = min(
                            int(env_current_episode_step[env_i].item()) + 1,
                            self.max_episode_steps - 1,
                        )

            with torch.no_grad():
                start = torch.clip(env_current_episode_step - self.trxl_memory_length, 0)
                end = torch.clip(env_current_episode_step, self.trxl_memory_length)
                bootstrap_indices = torch.stack(
                    [
                        torch.arange(start[b], end[b], device=self.device)
                        for b in range(self.num_envs)
                    ]
                ).long()
                bootstrap_window = batched_index_select(next_memory, 1, bootstrap_indices)
                next_value = self.agent.get_value(
                    next_obs,
                    bootstrap_window,
                    self.memory_mask[
                        torch.clip(env_current_episode_step, 0, self.trxl_memory_length - 1)
                    ],
                    stored_memory_indices[-1],
                )

                advantages = torch.zeros_like(rewards, device=self.device)
                lastgaelam = torch.zeros(self.num_envs, device=self.device)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values

            b_obs = obs.reshape(-1, *obs.shape[2:])
            b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
            b_actions = actions.reshape(-1, *actions.shape[2:])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_memory_index = stored_memory_index.reshape(-1)
            b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
            b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
            stored_memories_tensor = torch.stack(stored_memories, dim=0)

            actual_max_episode_steps = (
                (stored_memory_indices * stored_memory_masks.long()).max().item() + 1
            )
            if actual_max_episode_steps < self.trxl_memory_length:
                b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
                b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
                stored_memories_tensor = stored_memories_tensor[:, :actual_max_episode_steps]

            clipfracs: list[float] = []
            pg_loss = torch.tensor(0.0, device=self.device)
            v_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            r_loss = torch.tensor(0.0, device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            old_approx_kl = torch.tensor(0.0, device=self.device)
            approx_kl = torch.tensor(0.0, device=self.device)

            early_stop = False
            for _ in range(self.update_epochs):
                b_inds = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_memories = stored_memories_tensor[b_memory_index[mb_inds]]
                    mb_memory_windows = batched_index_select(
                        mb_memories,
                        1,
                        b_memory_indices[mb_inds],
                    )

                    _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        mb_memory_windows,
                        b_memory_mask[mb_inds],
                        b_memory_indices[mb_inds],
                        b_actions[mb_inds],
                    )

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
                    mb_advantages = mb_advantages.unsqueeze(1).repeat(
                        1,
                        len(self.action_space_shape),
                    )

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = torch.exp(logratio)
                    pgloss1 = -mb_advantages * ratio
                    pgloss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - self.clip_coef,
                        1.0 + self.clip_coef,
                    )
                    pg_loss = torch.max(pgloss1, pgloss2).mean()

                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    if self.clip_vloss:
                        v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                            min=-self.clip_coef,
                            max=self.clip_coef,
                        )
                        v_loss = torch.max(
                            v_loss_unclipped,
                            (v_loss_clipped - b_returns[mb_inds]) ** 2,
                        ).mean()
                    else:
                        v_loss = v_loss_unclipped.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * self.vf_coef

                    r_loss = torch.tensor(0.0, device=self.device)
                    if self.reconstruction_coef > 0.0 and self.agent.is_image_obs:
                        reconstruction = self.agent.reconstruct_observation()
                        r_loss = self.bce_loss(reconstruction, b_obs[mb_inds] / 255.0)
                        loss = loss + self.reconstruction_coef * r_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        )

                    if self.target_kl is not None and approx_kl.item() > self.target_kl:
                        early_stop = True
                        break

                if early_stop:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            return_mean = float(np.mean(self.recent_returns)) if self.recent_returns else float("nan")
            length_mean = float(np.mean(self.recent_lengths)) if self.recent_lengths else float("nan")
            sps = int(global_step / max(time.time() - start_time, 1e-6))

            progress_metrics = {
                "train/lr": float(lr),
                "train/ent_coef": float(ent_coef),
                "train/policy_loss": float(pg_loss.item()),
                "train/value_loss": float(v_loss.item()),
                "train/loss": float(loss.item()),
                "train/entropy": float(entropy_loss.item()),
                "train/reconstruction_loss": float(r_loss.item()),
                "train/old_approx_kl": float(old_approx_kl.item()),
                "train/approx_kl": float(approx_kl.item()),
                "train/clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
                "train/explained_variance": float(explained_var),
                "train/value_mean": float(values.mean().item()),
                "train/advantage_mean": float(advantages.mean().item()),
                "train/sps": float(sps),
            }

            if self.recent_returns:
                progress_metrics["train/episode_return_mean"] = return_mean
            if self.recent_lengths:
                progress_metrics["train/episode_length_mean"] = length_mean

            log_wandb(progress_metrics, step=global_step, silent=True)

            if self.recent_returns:
                print(
                    "[PPO-TRXL][progress] "
                    f"iter={iteration}/{num_iterations}, "
                    f"step={global_step}/{num_iterations * batch_size}, "
                    f"sps={sps}, "
                    f"return={return_mean:.3f}, "
                    f"length={length_mean:.1f}, "
                    f"pi_loss={pg_loss.item():.3f}, "
                    f"v_loss={v_loss.item():.3f}, "
                    f"entropy={entropy_loss.item():.3f}"
                )
            else:
                print(
                    "[PPO-TRXL][progress] "
                    f"iter={iteration}/{num_iterations}, "
                    f"step={global_step}/{num_iterations * batch_size}, "
                    f"sps={sps}, "
                    f"pi_loss={pg_loss.item():.3f}, "
                    f"v_loss={v_loss.item():.3f}, "
                    f"entropy={entropy_loss.item():.3f}"
                )

            if eval_interval > 0 and self.last_eval < global_step // eval_interval:
                self.last_eval = global_step // eval_interval
                eval_metrics = self.evaluate(
                    n_episodes=10,
                    seed=self.seed + 10_000 + self.last_eval,
                )
                log_wandb(eval_metrics, step=global_step, silent=True)
                print(
                    f"[PPO-TRXL][eval] step={global_step}: {_format_metrics(eval_metrics)}"
                )

            if save_interval > 0 and self.last_checkpoint < global_step // save_interval:
                self.last_checkpoint = global_step // save_interval
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, "ppo_trxl.pt")
                torch.save(
                    {
                        "model": self.agent.state_dict(),
                        "config": {
                            "env_id": self.env_id,
                            "trxl_num_layers": self.trxl_num_layers,
                            "trxl_num_heads": self.trxl_num_heads,
                            "trxl_dim": self.trxl_dim,
                            "trxl_memory_length": self.trxl_memory_length,
                            "trxl_positional_encoding": self.trxl_positional_encoding,
                            "max_episode_steps": self.max_episode_steps,
                            "action_space_shape": tuple(self.action_space_shape),
                            "obs_shape": tuple(self.obs_shape),
                        },
                    },
                    ckpt_path,
                )
                print(f"[PPO-TRXL][checkpoint] step={global_step}: saved {ckpt_path}")

        self.env.close()

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, seed: int = 42) -> Dict[str, float]:
        returns: list[float] = []

        self.agent.eval()

        eval_env = self._make_env(seed=seed, env_index=0, record_video=False)
        for ep in range(int(n_episodes)):
            obs, _ = eval_env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            t = 0

            memory = torch.zeros(
                (1, self.max_episode_steps, self.trxl_num_layers, self.trxl_dim),
                dtype=torch.float32,
                device=self.device,
            )

            while not done:
                idx = min(t, self.max_episode_steps - 1)
                indices = self.memory_indices[idx].unsqueeze(0)
                mask = self.memory_mask[min(idx, self.trxl_memory_length - 1)].unsqueeze(0)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                memory_window = batched_index_select(memory, 1, indices)

                action, _, _, _, new_memory = self.agent.get_action_and_value(
                    obs_t,
                    memory_window,
                    mask,
                    indices,
                    deterministic=True,
                )
                memory[:, idx] = new_memory

                action_np = action.cpu().numpy()
                if self._is_single_discrete:
                    action_exec = int(action_np.item())
                else:
                    action_exec = action_np.squeeze(0)

                obs, reward, terminated, truncated, _ = eval_env.step(action_exec)
                ep_return += float(reward)
                done = bool(terminated or truncated)
                t += 1

            returns.append(ep_return)

        eval_env.close()
        self.agent.train()

        returns_arr = np.asarray(returns, dtype=np.float32)
        return {
            "eval/return_mean": float(np.mean(returns_arr)),
            "eval/return_std": float(np.std(returns_arr)),
            "eval/return_min": float(np.min(returns_arr)),
            "eval/return_max": float(np.max(returns_arr)),
        }
