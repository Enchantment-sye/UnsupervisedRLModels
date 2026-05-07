from ..adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from ..base_spec import FloorGradientOverlaySpec, IsaacLabTaskSpec
from ..cfg_builders import humanoid_close_view_cfg_builder
from ..registry import register_task_spec


register_task_spec(
    IsaacLabTaskSpec(
        task_name="isaaclab_cartpole",
        env_id="Isaac-Cartpole-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        default_num_envs=1,
        notes="v1 stable",
    )
)

register_task_spec(
    IsaacLabTaskSpec(
        task_name="isaaclab_ant",
        env_id="Isaac-Ant-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        default_num_envs=1,
        notes="v1 stable",
    )
)

register_task_spec(
    IsaacLabTaskSpec(
        task_name="isaaclab_humanoid",
        env_id="Isaac-Humanoid-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        cfg_builder=humanoid_close_view_cfg_builder,
        default_num_envs=1,
        floor_gradient_overlay=FloorGradientOverlaySpec(
            preset_name="dmc_quadruped_run_forward_color",
        ),
        notes="v1 stable with gradient floor",
    )
)
