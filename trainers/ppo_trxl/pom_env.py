from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

POM_ENV_ID = "ProofofMemory-v0"
POM_MAX_EPISODE_STEPS = 16


class PoMEnv(gym.Env):
    """Proof-of-Concept memory environment used in CleanRL's TRxL example."""

    metadata = {
        "render_modes": ["human", "rgb_array", "debug_rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: str | None = None):
        self._freeze = True
        self._step_size = 0.2
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        num_steps = int(0.4 / self._step_size)
        lower = min(-2.0 * self._step_size, -num_steps * self._step_size)
        upper = max(3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size)
        positions = np.arange(lower, upper, self._step_size)
        positions = positions.clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = [round(float(x), 2) for x in positions]

        self.width = 400
        self.height = 80
        self.cell_width = self.width / (2 * int(1 / self._step_size) + 1)

        self._position = 0.0
        self._step_count = 0
        self._goals = np.array([-1.0, 1.0], dtype=np.float32)
        self.rewards: list[float] = []

    def step(self, action: int):
        reward = 0.0
        terminated = False

        if self._num_show_steps > self._step_count:
            delta = self._step_size * (1 - int(self._freeze))
            self._position += delta if int(action) == 1 else -delta
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)

            if self._freeze:
                self._step_count += 1
                return obs, reward, terminated, False, {}
        else:
            self._position += self._step_size if int(action) == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0], dtype=np.float32)

        if self._position == -1.0:
            goal_reward = 1.0 + self._min_steps * self._time_penalty
            reward += goal_reward if self._goals[0] == 1.0 else -goal_reward
            terminated = True
        elif self._position == 1.0:
            goal_reward = 1.0 + self._min_steps * self._time_penalty
            reward += goal_reward if self._goals[1] == 1.0 else -goal_reward
            terminated = True
        else:
            reward -= self._time_penalty

        self.rewards.append(float(reward))

        info = (
            {"reward": float(sum(self.rewards)), "length": len(self.rewards)}
            if terminated
            else {}
        )

        self._step_count += 1
        return obs, float(reward), terminated, False, info

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.rewards = []
        self._position = float(self.np_random.choice(self.possible_positions))
        self._step_count = 0

        goals = np.asarray([-1.0, 1.0], dtype=np.float32)
        self._goals = goals[self.np_random.permutation(2)]

        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs, {}

    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            return None

        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        num_cells = 2 * int(1 / self._step_size) + 1

        for i in range(num_cells):
            x = int(i * self.cell_width)
            canvas[:, max(0, x - 1) : min(self.width, x + 1)] = 200

        show_goals = self._num_show_steps > self._step_count
        if show_goals:
            left_color = np.array([0, 255, 0], dtype=np.uint8) if self._goals[0] > 0 else np.array([255, 0, 0], dtype=np.uint8)
            right_color = np.array([0, 255, 0], dtype=np.uint8) if self._goals[1] > 0 else np.array([255, 0, 0], dtype=np.uint8)
            canvas[:, : int(self.cell_width)] = left_color
            canvas[:, self.width - int(self.cell_width) :] = right_color
        else:
            canvas[:, : int(self.cell_width)] = 200
            canvas[:, self.width - int(self.cell_width) :] = 200

        agent_pos = int((self._position + 1) / self._step_size)
        agent_x = int(agent_pos * self.cell_width + self.cell_width / 2)
        agent_x = max(0, min(self.width - 1, agent_x))
        canvas[max(0, self.height // 2 - 8) : min(self.height, self.height // 2 + 8), max(0, agent_x - 8) : min(self.width, agent_x + 8)] = np.array([0, 0, 255], dtype=np.uint8)

        if self.render_mode in {"rgb_array", "debug_rgb_array"}:
            return canvas
        return None

    def close(self):
        return None



def register_pom_env() -> None:
    try:
        gym.spec(POM_ENV_ID)
        return
    except gym.error.Error:
        pass

    gym.register(
        id=POM_ENV_ID,
        entry_point="trainers.ppo_trxl.pom_env:PoMEnv",
        max_episode_steps=POM_MAX_EPISODE_STEPS,
    )

