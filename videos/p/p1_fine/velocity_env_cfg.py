import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from magiclab_rl_lab.assets.robots.magiclab import MAGICLAB_Z1_12DOF_CFG as ROBOT_CFG
from magiclab_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = None
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=[-0.1, 0.1], lin_vel_y=[-0.1, 0.1], ang_vel_z=[-0.1, 0.1]
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=[-0.1, 0.1], lin_vel_y=[-0.1, 0.1], ang_vel_z=[-0.1, 0.1]
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        scale=0.25, use_default_offset=True, 
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ], 
                                preserve_order=True)},
                                noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ], 
                                preserve_order=True)},
                                scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action,
                                clip=(-100.0, 100.0),
                                scale=1.0,
                              )
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.6})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg("robot",
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ],
                                preserve_order=True)},
                                )
        
        
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, 
                                params={"asset_cfg": SceneEntityCfg("robot", 
                                joint_names=[
                                    "left_hip_pitch_joint",
                                    "left_hip_roll_joint",
                                    "left_hip_yaw_joint",
                                    "left_knee_joint",
                                    "left_ankle_pitch_joint",
                                    "left_ankle_roll_joint",
                                    "right_hip_pitch_joint",
                                    "right_hip_roll_joint",
                                    "right_hip_yaw_joint",
                                    "right_knee_joint",
                                    "right_ankle_pitch_joint",
                                    "right_ankle_roll_joint",
                                ],
                                preserve_order=True)},
                                scale=0.05)
        last_action = ObsTerm(func=mdp.last_action,
                              clip=(-100.0, 100.0),
                              scale=1.0,
                              )
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.6})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )
        contact_mask = ObsTerm(func=mdp.contact_mask, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    alive = RewTerm(func=mdp.is_alive, weight=0.3)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-8.0, params={"target_height": 0.7})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.04)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    energy = RewTerm(func=mdp.energy, weight=-2e-05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    joint_deviation_legs = RewTerm(func=mdp.joint_deviation_l1, weight=-0.7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])})
    feet_contact_number = RewTerm(func=mdp.feet_contact_number, weight=0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "period": 0.6})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
    feet_clearance = RewTerm(func=mdp.foot_clearance_reward, weight=0.0, params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")})
    stand_still = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "command_name": "base_velocity", "command_threshold": 0.05})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=0.0, params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=16384, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt



@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # terrain_generator is None in plane mode
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # disable curriculum for play mode (avoid reward term lookup errors)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
