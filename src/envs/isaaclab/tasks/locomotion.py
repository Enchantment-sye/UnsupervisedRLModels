from ..adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from ..base_spec import FloorGradientOverlaySpec, IsaacLabTaskSpec
from ..cfg_builders import anymal_close_view_cfg_builder, humanoid_close_view_cfg_builder
from ..registry import register_task_spec


_GRADIENT = FloorGradientOverlaySpec(
    preset_name="dmc_quadruped_run_forward_color",
)


def _register_unitree_locomotion_task(task_name, env_id, *, robot_kind, terrain, play=False):
    cfg_builder = anymal_close_view_cfg_builder if robot_kind == "quadruped" else humanoid_close_view_cfg_builder
    notes = f"unitree {terrain} locomotion"
    if play:
        notes += " play"
    register_task_spec(
        IsaacLabTaskSpec(
            task_name=task_name,
            env_id=env_id,
            workflow_type="manager",
            obs_type="box",
            action_type="box",
            requires_cameras=False,
            supports_render_rgb=True,
            supports_camera_obs=True,
            camera_obs_key="rgb",
            adapter_cls=IsaacLabSingleAgentBoxEnv,
            cfg_builder=cfg_builder,
            default_num_envs=1,
            floor_gradient_overlay=_GRADIENT,
            notes=notes,
        )
    )


register_task_spec(
    IsaacLabTaskSpec(
        task_name="isaaclab_velocity_flat_anymal_c",
        env_id="Isaac-Velocity-Flat-Anymal-C-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        cfg_builder=anymal_close_view_cfg_builder,
        default_num_envs=1,
        floor_gradient_overlay=_GRADIENT,
        notes="v1 locomotion gradient floor",
    )
)

for _task_name, _env_id, _robot_kind, _terrain, _play in (
    ("isaaclab_velocity_flat_unitree_a1", "Isaac-Velocity-Flat-Unitree-A1-v0", "quadruped", "flat", False),
    ("isaaclab_velocity_flat_unitree_a1_play", "Isaac-Velocity-Flat-Unitree-A1-Play-v0", "quadruped", "flat", True),
    ("isaaclab_velocity_rough_unitree_a1", "Isaac-Velocity-Rough-Unitree-A1-v0", "quadruped", "rough", False),
    ("isaaclab_velocity_rough_unitree_a1_play", "Isaac-Velocity-Rough-Unitree-A1-Play-v0", "quadruped", "rough", True),
    ("isaaclab_velocity_flat_unitree_go1", "Isaac-Velocity-Flat-Unitree-Go1-v0", "quadruped", "flat", False),
    ("isaaclab_velocity_flat_unitree_go1_play", "Isaac-Velocity-Flat-Unitree-Go1-Play-v0", "quadruped", "flat", True),
    ("isaaclab_velocity_rough_unitree_go1", "Isaac-Velocity-Rough-Unitree-Go1-v0", "quadruped", "rough", False),
    ("isaaclab_velocity_rough_unitree_go1_play", "Isaac-Velocity-Rough-Unitree-Go1-Play-v0", "quadruped", "rough", True),
    ("isaaclab_velocity_flat_unitree_go2", "Isaac-Velocity-Flat-Unitree-Go2-v0", "quadruped", "flat", False),
    ("isaaclab_velocity_flat_unitree_go2_play", "Isaac-Velocity-Flat-Unitree-Go2-Play-v0", "quadruped", "flat", True),
    ("isaaclab_velocity_rough_unitree_go2", "Isaac-Velocity-Rough-Unitree-Go2-v0", "quadruped", "rough", False),
    ("isaaclab_velocity_rough_unitree_go2_play", "Isaac-Velocity-Rough-Unitree-Go2-Play-v0", "quadruped", "rough", True),
    ("isaaclab_velocity_flat_h1", "Isaac-Velocity-Flat-H1-v0", "humanoid", "flat", False),
    ("isaaclab_velocity_flat_h1_play", "Isaac-Velocity-Flat-H1-Play-v0", "humanoid", "flat", True),
    ("isaaclab_velocity_rough_h1", "Isaac-Velocity-Rough-H1-v0", "humanoid", "rough", False),
    ("isaaclab_velocity_rough_h1_play", "Isaac-Velocity-Rough-H1-Play-v0", "humanoid", "rough", True),
    ("isaaclab_velocity_flat_g1", "Isaac-Velocity-Flat-G1-v0", "humanoid", "flat", False),
    ("isaaclab_velocity_flat_g1_play", "Isaac-Velocity-Flat-G1-Play-v0", "humanoid", "flat", True),
    ("isaaclab_velocity_rough_g1", "Isaac-Velocity-Rough-G1-v0", "humanoid", "rough", False),
    ("isaaclab_velocity_rough_g1_play", "Isaac-Velocity-Rough-G1-Play-v0", "humanoid", "rough", True),
):
    _register_unitree_locomotion_task(
        _task_name,
        _env_id,
        robot_kind=_robot_kind,
        terrain=_terrain,
        play=_play,
    )
