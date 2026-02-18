"""Curriculum terms for standup training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from holosoma.managers.curriculum.base import CurriculumTermBase


class VerticalPullCurriculum(CurriculumTermBase):
    """Curriculum that ramps down a vertical pulling force as episodes get longer.

    Inverse of PenaltyCurriculum: longer episodes indicate the robot is learning
    to stand, so the assistive force is reduced. When episodes are short (robot
    falls quickly) the force stays high to maintain exploration.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)

        params = cfg.params or {}
        self.initial_scale = float(params.get("initial_scale", 1.0))
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 500.0))
        self.degree = float(params.get("degree", 0.0005))

        self.current_scale = self.initial_scale

    def setup(self) -> None:
        """Initialize the vertical pull scale on the environment."""
        self.env._vertical_pull_scale = self.current_scale

    def reset(self, env_ids) -> None:
        """Decrease pull force as average episode length grows."""
        average_length = float(self.env.average_episode_length)

        if average_length > self.level_up_threshold:
            self.current_scale *= 1.0 - self.degree

        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))
        self.env._vertical_pull_scale = self.current_scale

        if hasattr(self.env, "log_dict"):
            self.env.log_dict["vertical_pull_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        """Clamp pull scale within bounds each step."""
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))
        self.env._vertical_pull_scale = self.current_scale

    def state_dict(self) -> dict[str, Any]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        scale = state.get("current_scale")
        if scale is not None:
            self.current_scale = float(scale)
            self.env._vertical_pull_scale = self.current_scale
