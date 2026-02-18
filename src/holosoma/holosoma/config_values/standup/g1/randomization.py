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

g1_29dof_standup_randomization = replace(
    g1_29dof_randomization,
    step_terms=_standup_step_terms,
)

__all__ = ["g1_29dof_standup_randomization"]
