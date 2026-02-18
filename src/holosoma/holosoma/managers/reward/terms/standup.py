"""Reward terms for standup-from-supine tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def body_height_tracking(
    env: LeggedRobotLocomotionManager,
    *,
    body_names: list[str],
    target_height: float = 0.75,
    tracking_sigma: float = 0.1,
) -> torch.Tensor:
    """Exponential reward tracking average height of named bodies relative to feet.

    A generic version of :func:`shoulder_height_tracking` that accepts
    arbitrary body names, useful when the tracked reference points differ
    between robot morphologies or experimental setups.

    Args:
        env: The simulation environment.
        body_names: Names of bodies whose z-positions are averaged.
        target_height: Target height above average foot height in metres.
        tracking_sigma: Width of the exponential reward bell; smaller
            is tighter.

    Returns:
        Reward tensor of shape ``(num_envs,)``, values in ``[0, 1]``.
    """
    body_indices = [env.body_names.index(name) for name in body_names]
    body_z_values = [env.simulator._rigid_body_pos[:, idx, 2] for idx in body_indices]
    body_avg = torch.stack(body_z_values).mean(dim=0)

    left_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[0], 2]
    right_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[1], 2]
    feet_avg = (left_foot_z + right_foot_z) / 2.0

    current_height = torch.nan_to_num(body_avg - feet_avg, nan=0.0, posinf=2.0, neginf=0.0)
    error = torch.square(target_height - current_height)
    return torch.nan_to_num(torch.exp(-error / tracking_sigma), nan=0.0)


def shoulder_height_tracking(
    env: LeggedRobotLocomotionManager,
    *,
    target_height: float = 0.9,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for average shoulder height relative to feet.

    Measuring relative to feet makes the signal invariant to terrain
    elevation. The G1 at full standing height has a shoulder-to-feet
    distance of roughly 0.9 m.

    Args:
        env: The simulation environment.
        target_height: Target shoulder-to-feet distance in metres.
        tracking_sigma: Width of the exponential reward bell; smaller
            is tighter.

    Returns:
        Reward tensor of shape ``(num_envs,)``, values in ``[0, 1]``.
    """
    left_idx = env.body_names.index("left_shoulder_pitch_link")
    right_idx = env.body_names.index("right_shoulder_pitch_link")
    left_shoulder_z = env.simulator._rigid_body_pos[:, left_idx, 2]
    right_shoulder_z = env.simulator._rigid_body_pos[:, right_idx, 2]
    shoulder_avg_z = (left_shoulder_z + right_shoulder_z) / 2.0

    left_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[0], 2]
    right_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[1], 2]
    feet_avg_z = (left_foot_z + right_foot_z) / 2.0

    current_height = shoulder_avg_z - feet_avg_z
    current_height = torch.nan_to_num(current_height, nan=0.0, posinf=2.0, neginf=0.0)

    error = torch.square(target_height - current_height)
    reward = torch.exp(-error / tracking_sigma)
    return torch.nan_to_num(reward, nan=0.0)


def uprightness_height_gated(
    env: LeggedRobotLocomotionManager,
    *,
    height_threshold: float = 0.5,
) -> torch.Tensor:
    """Uprightness penalty zeroed below a shoulder height gate.

    Suppresses the penalty while the robot is near the ground so the
    policy can explore recovery strategies without being punished for
    the tilt that is necessary during standup. ~0.5 m shoulder-to-feet
    corresponds roughly to a kneeling posture.

    Args:
        env: The simulation environment.
        height_threshold: Shoulder-to-feet height (m) below which the
            penalty is zeroed.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    projected = get_projected_gravity(env)
    base_penalty = torch.sum(torch.square(projected[:, :2]), dim=1)

    left_idx = env.body_names.index("left_shoulder_pitch_link")
    right_idx = env.body_names.index("right_shoulder_pitch_link")
    left_z = env.simulator._rigid_body_pos[:, left_idx, 2]
    right_z = env.simulator._rigid_body_pos[:, right_idx, 2]
    shoulder_avg_z = (left_z + right_z) / 2.0

    left_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[0], 2]
    right_foot_z = env.simulator._rigid_body_pos[:, env.feet_indices[1], 2]
    feet_avg_z = (left_foot_z + right_foot_z) / 2.0

    current_height = torch.nan_to_num(shoulder_avg_z - feet_avg_z, nan=0.0)
    gate = (current_height >= height_threshold).float()
    return base_penalty * gate


def non_foot_contact_penalty(
    env: LeggedRobotLocomotionManager,
    *,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Count of non-foot bodies with contact force above threshold.

    Discourages resting weight on knees, shins, or hands. Non-foot body
    indices are computed once and cached on the env object.

    Args:
        env: The simulation environment.
        force_threshold: Contact force magnitude (N) above which a body
            counts as in undesired contact.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    if not hasattr(env, "_standup_non_foot_body_indices"):
        feet_set = set(env.feet_indices.tolist())
        non_foot = [i for i in range(len(env.body_names)) if i not in feet_set]
        env._standup_non_foot_body_indices = torch.tensor(non_foot, dtype=torch.long, device=env.device)  # ty: ignore[invalid-assignment]

    forces = env.simulator.contact_forces[:, env._standup_non_foot_body_indices, :]  # ty: ignore[unresolved-attribute]
    is_contact = torch.norm(forces, dim=-1) > force_threshold
    return torch.sum(is_contact.float(), dim=1)


def xy_drift_penalty(
    env: LeggedRobotLocomotionManager,
    *,
    max_drift: float = 0.75,
) -> torch.Tensor:
    """Quadratic penalty for XY displacement from episode start position.

    Start position is recorded on first call and refreshed after each
    reset (when ``episode_length_buf == 0``).

    Args:
        env: The simulation environment.
        max_drift: Distance (m) at which the penalty saturates to 1.0.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    current_xy = env.simulator.robot_root_states[:, :2]

    if not hasattr(env, "_standup_start_xy"):
        env._standup_start_xy = current_xy.clone()  # ty: ignore[invalid-assignment]

    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._standup_start_xy[reset_mask] = current_xy[reset_mask].clone()  # ty: ignore[unresolved-attribute]

    displacement = torch.norm(current_xy - env._standup_start_xy, dim=1)  # ty: ignore[unresolved-attribute]
    return torch.square(displacement / max_drift)


def kinetic_energy_penalty(
    env: LeggedRobotLocomotionManager,
    *,
    velocity_scale: float = 10.0,
) -> torch.Tensor:
    """Sum of squared linear velocities of non-foot bodies, scaled.

    Penalises fast-moving body parts as a proxy for impact risk. Non-foot
    body indices are computed once and cached on the env object.

    Args:
        env: The simulation environment.
        velocity_scale: Divisor applied to the raw Σ‖v‖² value (m²/s²).
            Larger values produce a softer penalty.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    if not hasattr(env, "_standup_ke_non_foot_indices"):
        feet_set = set(env.feet_indices.tolist())
        non_foot = [i for i in range(len(env.body_names)) if i not in feet_set]
        env._standup_ke_non_foot_indices = torch.tensor(non_foot, dtype=torch.long, device=env.device)  # ty: ignore[invalid-assignment]

    vel = env.simulator._rigid_body_vel[:, env._standup_ke_non_foot_indices, :]  # ty: ignore[unresolved-attribute]
    return vel.square().sum(dim=(1, 2)) / velocity_scale
