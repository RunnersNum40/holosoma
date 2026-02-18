"""Termination conditions for standup-from-prone tasks."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def floor_penetration(env, *, threshold: float = -0.3) -> torch.Tensor:
    """Terminate if the root body drops below a z-height threshold.

    For standup training the threshold should be negative (below the floor),
    catching cases where the physics solver has allowed the robot to clip
    through the terrain rather than genuine falls.  Normal standup motion
    keeps the root well above zero.

    Args:
        env: The simulation environment.
        threshold: Minimum allowable root z-position in metres.  The default
            of -0.3 triggers only on clear geometry penetration.

    Returns:
        Boolean termination tensor of shape ``(num_envs,)``.
    """
    root_z = env.simulator.robot_root_states[:, 2]
    return root_z < threshold
