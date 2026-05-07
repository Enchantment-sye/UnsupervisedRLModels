from ..adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from ..base_spec import IsaacLabTaskSpec
from ..registry import register_task_spec


register_task_spec(
    IsaacLabTaskSpec(
        task_name="isaaclab_quadcopter",
        env_id="Isaac-Quadcopter-Direct-v0",
        workflow_type="direct",
        obs_type="box",
        action_type="box",
        requires_cameras=False,
        supports_render_rgb=True,
        supports_camera_obs=True,
        camera_obs_key="rgb",
        adapter_cls=IsaacLabSingleAgentBoxEnv,
        default_num_envs=1,
        notes="phase-2 placeholder",
    )
)
