"""Standup curriculum presets for the G1 robot.

Includes the standard penalty curriculum plus a vertical pull curriculum
that ramps down the assistive force as the robot learns to stand.
"""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_standup_discovery_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.00025,
            },
        ),
        "vertical_pull_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.standup:VerticalPullCurriculum",
            params={
                "initial_scale": 1.0,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_up_threshold": 500.0,
                "degree": 0.0005,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["g1_29dof_standup_discovery_curriculum"]
