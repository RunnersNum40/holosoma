"""Standup termination presets for the G1 29-DOF robot.

Contact termination is omitted: the robot starts on the floor and must
use non-foot body contact during recovery. The non_foot_contact_penalty
reward term discourages prolonged contact without hard episode termination.
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
