"""Randomization terms for standup training."""

from __future__ import annotations

import torch


def apply_vertical_pull(
    env,
    *,
    max_force: float = 206.0,
    enabled: bool = True,
    **_,
) -> None:
    """Apply an upward pulling force on the torso to aid standup discovery.

    Inspired by the HoST paper: a vertical force (~60% body weight) is applied
    to the torso and curricularized down to zero as the robot learns to stand.

    The actual force magnitude is ``max_force * env._vertical_pull_scale``,
    where ``_vertical_pull_scale`` is managed by ``VerticalPullCurriculum``.

    Args:
        max_force: Peak upward force in Newtons (default ~0.6 * 35kg * 9.81).
        enabled: Master toggle for this term.
    """
    if not enabled or env.is_evaluating:
        force = 0.0
    else:
        scale = getattr(env, "_vertical_pull_scale", 1.0)
        force = max_force * scale

    torso_idx = env.torso_index
    sim = env.simulator

    if hasattr(sim, "applied_forces"):
        applied = sim.applied_forces
        if isinstance(applied, torch.Tensor):
            applied[:, torso_idx, 2] = force
        else:
            applied[torso_idx, 2] = force
    else:
        num_envs = sim.num_envs
        body_id = sim.body_ids[torso_idx]
        forces = torch.zeros(num_envs, 1, 3, device=sim.sim_device)
        forces[:, 0, 2] = force
        torques = torch.zeros_like(forces)
        sim._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=torch.tensor([body_id], device=sim.sim_device),
        )


def clear_applied_forces(
    env,
    env_ids: torch.Tensor,
    **_,
) -> None:
    """Zero applied forces for reset environments.

    Prevents stale ``apply_vertical_pull`` forces from persisting across
    episode boundaries. Without this, a freshly spawned supine robot receives
    the full pull force on its first physics step while already clipping the
    ground, causing explosive contact resolution.

    Args:
        env: The simulation environment.
        env_ids: Indices of environments being reset.
    """
    sim = env.simulator

    if hasattr(sim, "applied_forces"):
        # MuJoCo path: direct array write
        applied = sim.applied_forces
        if isinstance(applied, torch.Tensor):
            applied[env_ids, :, :] = 0.0
        else:
            applied[:, :] = 0.0
    else:
        # IsaacSim path: zero out external forces for reset envs
        num_bodies = len(sim.body_ids)
        forces = torch.zeros(len(env_ids), num_bodies, 3, device=sim.sim_device)
        torques = torch.zeros_like(forces)
        sim._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=env_ids,
            body_ids=torch.tensor(sim.body_ids, device=sim.sim_device),
        )
