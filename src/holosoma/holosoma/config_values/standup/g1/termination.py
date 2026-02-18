"""Standup termination presets for the G1 robot.

Contact termination is intentionally disabled for standup: the robot starts
on the floor and must contact non-foot bodies (pelvis, knees, hands) during
the recovery.  The penalty-based ``non_foot_contact_penalty`` reward term
handles discouraging prolonged contact without hard-terminating episodes.

Termination conditions:
    timeout         — episode runs to maximum length (non-fatal timeout)
    floor_below_z   — root drops through the terrain (usually geometry error)
"""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_standup_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "floor_below_z": TerminationTermCfg(
            func="holosoma.managers.termination.terms.standup:floor_penetration",
            params={"threshold": -0.3},
        ),
    }
)

__all__ = ["g1_29dof_standup_termination"]
