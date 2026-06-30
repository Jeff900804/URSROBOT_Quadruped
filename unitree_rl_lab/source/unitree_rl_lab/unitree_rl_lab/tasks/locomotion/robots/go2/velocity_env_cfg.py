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

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from isaaclab.envs.manager_based_env import ManagerBasedEnv
import torch
import torch.nn.functional as F

# COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=0.0,
#     num_rows=10,
#     num_cols=10,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.0, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.0
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
#         ),
#         "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.0, grid_width=0.45, grid_height_range=(0.01, 0.5), platform_width=2.0
#         ),
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.05, 0.25),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.0,
#             step_height_range=(0.05, 0.25),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "Gap": terrain_gen.MeshGapTerrainCfg(
#             proportion=0.0,
#             gap_width_range=(0.1, 0.3),
#             platform_width=3.0,
#         ),
#         "stepping_stone": terrain_gen.HfSteppingStonesTerrainCfg(
#             proportion=0.0,
#             size=(10.0, 10.0),
#             stone_width_range=(0.5, 1.0),
#             stone_distance_range=(0.1, 0.3),
#             stone_height_max=0.0,
#             platform_width=2.0,
#             border_width=0.25,
#             holes_depth=-1.0  
#         ),
#         "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
#             proportion=1.0,
#             ring_thickness=1.0,
#             ring_width_range=(0.1, 0.3),
#             ring_height_range=(0.25, 0.45),
#             platform_width=2.0,
#         ), 
#         "Pit": terrain_gen.MeshPitTerrainCfg(
#             proportion=0.0,
#             pit_depth_range=(0.1, 0.3),
#             platform_width=3.0,
#         ),
#     },
# )

# 真的拿來走的：純平地
PHYS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.01, 0.05), platform_width=2.0
        ),
    },
)

# PHYS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=0.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         "stepping_stone": terrain_gen.HfSteppingStonesTerrainCfg(
#             proportion=1.0,
#             size=(10.0, 10.0),
#             stone_width_range=(0.5, 1.0),     
#             stone_distance_range=(0.1, 0.3), 
#             stone_height_max=0.05,
#             platform_width=2.0,
#             border_width=0.25,
#             holes_depth=-1.0,           
#         ),
#     },
# )


# # 拿來掃描風險的：stepping stones + holes
# RISK_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=0.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         "stepping_stone": terrain_gen.HfSteppingStonesTerrainCfg(
#             proportion=1.0,
#             size=(10.0, 10.0),
#             stone_width_range=(0.3, 1.0),
#             stone_distance_range=(0.1, 0.3),
#             stone_height_max=0.0,
#             platform_width=2.0,
#             border_width=0.25,
#             holes_depth=-1.0, 
#         ),
#     },
# )

# 拿來掃描風險的：floating_ring
RISK_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "stepping_stone": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.1,
            size=(10.0, 10.0),
            stone_width_range=(0.3, 1.0),
            stone_distance_range=(0.1, 0.3),
            stone_height_max=0.0,
            platform_width=2.0,
            border_width=0.25,
            holes_depth=-1.0, 
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.05), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "Gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.1,
            gap_width_range=(0.1, 0.3),
            platform_width=3.0,
        ),
    },
)


# @configclass
# class RobotSceneCfg(InteractiveSceneCfg):
#     """Configuration for the terrain scene with a legged robot."""

#     # ground terrain
#     terrain = TerrainImporterCfg(
#         prim_path="/World/ground",
#         terrain_type="generator",  # "plane", "generator"
#         terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
#         max_init_terrain_level=1,
#         collision_group=-1,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#         ),
#         visual_material=sim_utils.MdlFileCfg(
#             mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
#             project_uvw=True,
#             texture_scale=(0.25, 0.25),
#         ),
#         debug_vis=False,
#     )
#     # robots
#     robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#     # sensors
#     height_scanner = RayCasterCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/base",
#         offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
#         ray_alignment="yaw",
#         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
#         debug_vis=False,
#         mesh_prim_paths=["/World/ground"],
#     )
#     contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
#     # lights
#     sky_light = AssetBaseCfg(
#         prim_path="/World/skyLight",
#         spawn=sim_utils.DomeLightCfg(
#             intensity=750.0,
#             texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
#         ),
#     )

@configclass
class RobotSceneCfg(InteractiveSceneCfg):

    # 1) 先放 risk terrain（只給感知用）
    terrain_risk = TerrainImporterCfg(
        prim_path="/World/ground_risk",
        terrain_type="generator",
        terrain_generator=RISK_TERRAIN_CFG,
        collision_group=-1,     # 這裡先照填，真正禁用 collision 下面用事件做
        debug_vis=False,
        visual_material=None,   # 可選：不想看到它
    )

    # 2) 再放 physical terrain（真的拿來走）
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PHYS_TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground_risk"],  # ← 改這裡
        # mesh_prim_paths=["/World/ground"],  # ← 改這裡
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

from pxr import UsdGeom, UsdPhysics, Usd ,PhysxSchema

def disable_collision_and_hide_prim(env, env_ids, prim_path: str, hide: bool = False):
    stage = env.scene.stage if hasattr(env, "scene") and hasattr(env.scene, "stage") else None
    if stage is None:
        # 退而求其次：用 Isaac Sim 的 stage getter（看你檔案裡怎麼 import）
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"[disable_collision_and_hide_prim] prim not found: {prim_path}")

    # 可選：隱藏（我建議先 hide=False，避免 ray cast 被影響）
    if hide:
        img = UsdGeom.Imageable(root)
        if img:
            img.MakeInvisible()

    # 遞迴把底下所有 prim 的 USD Collision 關掉
    for prim in Usd.PrimRange(root):
        # 只要有可能碰撞的 prim，都套用/取得 CollisionAPI
        col_api = UsdPhysics.CollisionAPI.Apply(prim)
        # 有些 prim Apply 了也不一定真的會參與碰撞，但這樣最保險
        col_api.GetCollisionEnabledAttr().Set(False)


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup", 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
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
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # # Stepping stone
    disable_risk_collision = EventTerm(
        func=disable_collision_and_hide_prim,
        mode="startup",
        params={"prim_path": "/World/ground_risk"},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
        # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        # ),
        # limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        # ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        
        height_scanner = ObsTerm(func=mdp.height_scan,
           params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset":0.3},
           clip=(-10.0, 10.0),
        )
        
        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan,
           params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset":0.3},
           clip=(-10.0, 10.0),
        )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) #1.5
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.09)}
    ) #0.75

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0) #-2.0
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) #-0.05
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001) #-0.001
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4) #-2e-4
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05) #-0.1
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0) #-10.0
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5) #-2.5

    joint_pos = RewTerm(
        #func=mdp.joint_position_penalty_crouch,
        func=mdp.joint_position_penalty,
        weight=-0.3, #-0.7
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.0, #5.0
            "velocity_threshold": 0.1, #0.3
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1, #0.1
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,#-1.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1, #-0.1
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5, #-1.0
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # hole_td = RewTerm(
    #     func=mdp.hole_touchdown_penalty,
    #     params={
    #         "height_sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "grid_shape": (11, 17),
    #         "size_xy": (1.6, 1.0),
    #         "offset_z": 0.3,
    #         "hole_th": -0.05,          # 先用 0.0 讓它比較容易觸發
    #         "power": 1.0,
    #         "contact_force_th": 5.0,
    #         "event_penalty": 5.0,   # 先大一點，之後再調
    #         "command_name": "base_velocity",
    #         "min_cmd_x": 0.2,
    #     },
    #     weight=0.5,   # ✅ 函式本身回傳負值，所以 weight 用正的
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**28  # 268,435,456 (>= ~200M)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
