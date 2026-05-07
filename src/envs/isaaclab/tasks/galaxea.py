from ..adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from ..base_spec import IsaacLabTaskSpec
from ..cfg_builders import galaxea_workstation_cfg_builder
from ..registry import register_task_spec


def _register_galaxea_task(task_name, env_id, *, camera_key, notes, aliases=()):
    register_task_spec(
        IsaacLabTaskSpec(
            task_name=task_name,
            env_id=env_id,
            workflow_type="direct",
            obs_type="box",
            action_type="box",
            requires_cameras=False,
            supports_render_rgb=True,
            supports_camera_obs=True,
            camera_obs_key=camera_key,
            adapter_cls=IsaacLabSingleAgentBoxEnv,
            cfg_builder=galaxea_workstation_cfg_builder,
            default_num_envs=1,
            default_image_source_encoder0="render",
            default_image_source_encoder1="camera",
            aliases=tuple(aliases),
            notes=notes,
        )
    )


_register_galaxea_task(
    "isaaclab_r1_lift_bin",
    "Isaac-R1-Lift-Bin-IK-Rel-Direct-v0",
    camera_key="front_rgb",
    notes="Galaxea R1 fixed-workstation lift-bin baseline sourced from local Galaxea_Lab overlay",
)

_register_galaxea_task(
    "isaaclab_r1_multi_fruit",
    "Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",
    camera_key="front_rgb",
    notes="Galaxea R1 fixed-workstation multi-fruit manipulation baseline sourced from local Galaxea_Lab overlay",
)
