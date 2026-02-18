"""Standup randomization presets for the G1 robot.

Extends the standard locomotion randomization with a vertical pull force
and disables random pushes (counterproductive while learning to stand).
"""

from dataclasses import replace

from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization

_standup_step_terms = dict(g1_29dof_randomization.step_terms)
_standup_step_terms.pop("apply_pushes", None)
_standup_step_terms["apply_vertical_pull"] = RandomizationTermCfg(
    func="holosoma.managers.randomization.terms.standup:apply_vertical_pull",
    params={
        "max_force": 206.0,
        "enabled": True,
    },
)

_standup_reset_terms = dict(g1_29dof_randomization.reset_terms)
_standup_reset_terms["clear_applied_forces"] = RandomizationTermCfg(
    func="holosoma.managers.randomization.terms.standup:clear_applied_forces",
)

g1_29dof_standup_randomization = replace(
    g1_29dof_randomization,
    step_terms=_standup_step_terms,
    reset_terms=_standup_reset_terms,
)

__all__ = ["g1_29dof_standup_randomization"]
