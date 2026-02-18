"""Reward terms for standup-from-prone tasks.

These terms are designed for training a policy to stand up from a supine
(lying face-up) pose.  They complement the locomotion reward terms in
``holosoma.managers.reward.terms.locomotion`` with standup-specific signals.

G1 body name reference:
    torso_link             — upper torso (body 17)
    left_shoulder_pitch_link  — left shoulder pivot (body 18), ~1.0 m when standing
    right_shoulder_pitch_link — right shoulder pivot (body 25), ~1.0 m when standing
    left_foot_contact_point   — left foot (body 7)
    right_foot_contact_point  — right foot (body 14)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


def shoulder_height_tracking(
    env: BaseTask,
    *,
    target_height: float = 0.9,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for tracking average shoulder height relative to feet.

    Uses the average z-position of the left and right shoulder pitch links
    minus the average foot contact point z-position.  Measuring relative to
    feet makes the signal invariant to terrain elevation and foot placement.

    A G1 at full standing height has shoulder-to-feet distance of roughly
    0.9 m; this is the natural default target.

    Args:
        env: The simulation environment.
        target_height: Target shoulder-to-feet height in metres.
        tracking_sigma: Width of the exponential reward bell — smaller is
            tighter, e.g. 0.1 is precise, 0.4 is lenient.

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
    env: BaseTask,
    *,
    height_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize non-upright base orientation, suppressed near the ground.

    During the early phase of standup the robot is necessarily tilted;
    penalising tilt immediately discourages the policy from attempting the
    transition at all.  This gate zeros the uprightness penalty while the
    shoulder-to-feet height is below ``height_threshold``, freeing the policy
    to explore ground-contact strategies without orientation punishment.

    G1 reference heights (shoulder-to-feet):
        ~0.15 m — supine on the floor
        ~0.50 m — beginning to push up / kneeling
        ~0.90 m — fully upright

    Args:
        env: The simulation environment.
        height_threshold: Shoulder-to-feet height (m) below which the
            uprightness penalty is zeroed.

    Returns:
        Reward tensor of shape ``(num_envs,)`` — use with a negative weight.
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
    env: BaseTask,
    *,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contact forces on body parts other than the feet.

    Discourages the policy from resting weight on knees, shins, or hands as
    a crutch while getting up.  Returns the count of non-foot bodies whose
    contact force magnitude exceeds ``force_threshold`` Newtons.

    The set of non-foot body indices is built once and cached on the env
    object to avoid repeated set operations each step.

    Args:
        env: The simulation environment.
        force_threshold: Contact force magnitude (N) above which a body is
            counted as in undesired contact.

    Returns:
        Reward tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    if not hasattr(env, "_standup_non_foot_body_indices"):
        feet_set = set(env.feet_indices.tolist())
        non_foot = [i for i in range(len(env.body_names)) if i not in feet_set]
        env._standup_non_foot_body_indices = torch.tensor(non_foot, dtype=torch.long, device=env.device)

    forces = env.simulator.contact_forces[:, env._standup_non_foot_body_indices, :]
    is_contact = torch.norm(forces, dim=-1) > force_threshold
    return torch.sum(is_contact.float(), dim=1)


def xy_drift_penalty(
    env: BaseTask,
    *,
    max_drift: float = 0.75,
) -> torch.Tensor:
    """Penalize XY drift from the episode start position.

    The robot should stand up in place rather than sliding across the floor.
    The penalty is quadratic, normalised so it equals 1.0 when the robot has
    moved ``max_drift`` metres from its starting XY position.

    The start position is recorded at the first call to this function for
    each environment and refreshed whenever ``episode_length_buf == 0``
    (i.e., immediately after a reset).

    Args:
        env: The simulation environment.
        max_drift: Distance (m) at which the penalty saturates to 1.0.

    Returns:
        Reward tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    current_xy = env.simulator.robot_root_states[:, :2]

    if not hasattr(env, "_standup_start_xy"):
        env._standup_start_xy = current_xy.clone()

    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._standup_start_xy[reset_mask] = current_xy[reset_mask].clone()

    displacement = torch.norm(current_xy - env._standup_start_xy, dim=1)
    return torch.square(displacement / max_drift)


def kinetic_energy_penalty(
    env: BaseTask,
    *,
    velocity_scale: float = 10.0,
) -> torch.Tensor:
    """Penalize kinetic energy of non-foot rigid bodies.

    Computes the sum of squared linear velocities across all non-foot bodies.
    This is a simulator-independent proxy for kinetic energy that penalizes
    fast-moving body parts — the precondition for hard impacts.  Unlike
    contact-force penalties, this signal cannot be gamed by the soft contact
    model or time-averaging in the physics engine.

    The raw ``Σ ‖v‖²`` (m²/s²) is divided by ``velocity_scale`` to keep
    the reward term on a manageable numerical scale.

    Args:
        env: The simulation environment.
        velocity_scale: Divisor for raw sum-of-squared velocities (m²/s²).
            Larger values produce a softer penalty.

    Returns:
        Reward tensor of shape ``(num_envs,)`` — use with a negative weight.
    """
    if not hasattr(env, "_standup_ke_non_foot_indices"):
        feet_set = set(env.feet_indices.tolist())
        non_foot = [i for i in range(len(env.body_names)) if i not in feet_set]
        env._standup_ke_non_foot_indices = torch.tensor(non_foot, dtype=torch.long, device=env.device)

    vel = env.simulator._rigid_body_vel[:, env._standup_ke_non_foot_indices, :]
    return vel.square().sum(dim=(1, 2)) / velocity_scale
