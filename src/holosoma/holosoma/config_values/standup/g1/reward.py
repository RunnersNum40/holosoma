"""Standup reward presets for the G1 robot.

These configs are designed to be used with PPO and the LeggedRobotLocomotionManager
for training a policy to stand up from a supine pose.

Reward architecture:
    shoulder_height  — primary task signal; exponential tracking of shoulder-to-feet height
    uprightness      — orientation penalty, height-gated to avoid punishing ground phases
    action_rate      — smoothness regularisation
    torques          — energy/torque penalty
    non_foot_contact — discourages using knees/hands as props
    xy_drift         — keeps the robot standing in place
    kinetic_energy   — penalises ballistic motion and high-velocity body parts
    alive            — constant survival bonus
"""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_standup_ppo = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # ----------------------------------------------------------------
        # Primary task: reach and hold standing height
        # ----------------------------------------------------------------
        "shoulder_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:shoulder_height_tracking",
            weight=15.0,
            params={
                "target_height": 0.9,
                "tracking_sigma": 0.25,
            },
        ),
        # ----------------------------------------------------------------
        # Orientation: penalise tilt once robot has started to rise
        # ----------------------------------------------------------------
        "uprightness": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:uprightness_height_gated",
            weight=-8.0,
            params={"height_threshold": 0.5},
            tags=["penalty_curriculum"],
        ),
        # ----------------------------------------------------------------
        # Regularisation: smooth actions and low torques
        # ----------------------------------------------------------------
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
        # ----------------------------------------------------------------
        # Contact quality: no leaning on knees, shins, or hands
        # ----------------------------------------------------------------
        "non_foot_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:non_foot_contact_penalty",
            weight=-2.0,
            params={"force_threshold": 1.0},
            tags=["penalty_curriculum"],
        ),
        # ----------------------------------------------------------------
        # Positional stability: stand up in place, don't slide away
        # ----------------------------------------------------------------
        "xy_drift": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:xy_drift_penalty",
            weight=-2.0,
            params={"max_drift": 0.75},
        ),
        # ----------------------------------------------------------------
        # Kinetic energy: discourage thrashing and hard body impacts
        # ----------------------------------------------------------------
        "kinetic_energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.standup:kinetic_energy_penalty",
            weight=-2.0,
            params={"velocity_scale": 10.0},
            tags=["penalty_curriculum"],
        ),
        # ----------------------------------------------------------------
        # Survival bonus
        # ----------------------------------------------------------------
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=2.0,
        ),
    },
)

__all__ = ["g1_29dof_standup_ppo"]
