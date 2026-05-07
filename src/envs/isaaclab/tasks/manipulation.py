from ..adapters.single_agent_box import IsaacLabSingleAgentBoxEnv
from ..base_spec import IsaacLabTaskSpec
from ..registry import register_task_spec


def _register_manipulation_task(task_name, env_id, *, notes):
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
            default_num_envs=1,
            notes=notes,
        )
    )


_register_manipulation_task(
    "isaaclab_reach_franka",
    "Isaac-Reach-Franka-v0",
    notes="v1 stable",
)

_register_manipulation_task(
    "isaaclab_lift_cube_franka",
    "Isaac-Lift-Cube-Franka-v0",
    notes="phase-2 placeholder",
)

for _task_name, _env_id, _notes in (
    ("isaaclab_pick_place_g1_inspire_ftp", "Isaac-PickPlace-G1-InspireFTP-Abs-v0", "g1 pick-place with inspire ftp hand"),
    ("isaaclab_pick_place_locomanipulation_g1", "Isaac-PickPlace-Locomanipulation-G1-Abs-v0", "g1 locomanipulation pick-place"),
    ("isaaclab_pick_place_fixedbase_upperbodyik_g1", "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0", "g1 fixed-base upper-body ik pick-place"),
):
    _register_manipulation_task(_task_name, _env_id, notes=_notes)
