"""Termination conditions for standup-from-supine tasks."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def floor_penetration(env, *, threshold: float = -0.3) -> torch.Tensor:
    """Terminate if the root body z-position drops below a threshold.

    A negative threshold catches geometry penetration without triggering
    on normal standup motion.

    Args:
        env: The simulation environment.
        threshold: Minimum allowable root z-position in metres.

    Returns:
        Boolean termination tensor of shape ``(num_envs,)``.
    """
    root_z = env.simulator.robot_root_states[:, 2]
    return root_z < threshold
