"""Standup reward presets for the G1 29-DOF robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_standup_ppo = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "shoulder_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:shoulder_height_tracking",
            weight=15.0,
            params={
                "target_height": 0.9,
                "tracking_sigma": 0.25,
            },
        ),
        "uprightness": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:uprightness_height_gated",
            weight=-8.0,
            params={"height_threshold": 0.5},
            tags=["penalty_curriculum"],
        ),
        "action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-3.0,
            tags=["penalty_curriculum"],
        ),
        "torques": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_torques",
            weight=-1.0,
            tags=["penalty_curriculum"],
        ),
        "non_foot_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:non_foot_contact_penalty",
            weight=-2.0,
            params={"force_threshold": 1.0},
            tags=["penalty_curriculum"],
        ),
        "xy_drift": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:xy_drift_penalty",
            weight=-2.0,
            params={"max_drift": 0.75},
        ),
        "kinetic_energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:kinetic_energy_penalty",
            weight=-2.0,
            params={"velocity_scale": 10.0},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=2.0,
        ),
    },
)

__all__ = ["g1_29dof_standup_ppo"]
