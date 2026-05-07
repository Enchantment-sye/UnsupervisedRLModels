from __future__ import annotations


PRETRAIN_STAGE = "pre_training"
FINETUNE_STAGE = "finetune"
ZERO_TRAIN_STAGE = "zero_training"
VALID_STAGES = {PRETRAIN_STAGE, FINETUNE_STAGE, ZERO_TRAIN_STAGE}
CASCADE_ALGO_SUFFIX = "_cascade"
BASE_ALGO_NAMES = ("metra", "dads", "diayn", "lsd", "iksd", "csd", "idk_csd")
CASCADE_ALIAS_ALGO_NAMES = tuple(f"{algo}{CASCADE_ALGO_SUFFIX}" for algo in BASE_ALGO_NAMES)
SUPPORTED_ALGO_NAMES = BASE_ALGO_NAMES + CASCADE_ALIAS_ALGO_NAMES


def get_stage(config_or_stage) -> str:
    if isinstance(config_or_stage, str):
        stage = config_or_stage
    else:
        stage = getattr(config_or_stage, "stage", None)
        if stage is None and hasattr(config_or_stage, "log"):
            stage = getattr(config_or_stage.log, "stage", None)
    if stage not in VALID_STAGES:
        raise ValueError(f"Unsupported stage: {stage!r}")
    return stage


def is_pretraining_stage(config_or_stage) -> bool:
    return get_stage(config_or_stage) == PRETRAIN_STAGE


def is_finetune_stage(config_or_stage) -> bool:
    return get_stage(config_or_stage) == FINETUNE_STAGE


def is_zero_training_stage(config_or_stage) -> bool:
    return get_stage(config_or_stage) == ZERO_TRAIN_STAGE


def is_downstream_stage(config_or_stage) -> bool:
    return not is_pretraining_stage(config_or_stage)


def uses_intrinsic_reward(config_or_stage) -> bool:
    return is_pretraining_stage(config_or_stage)


def uses_external_reward(config_or_stage) -> bool:
    return is_downstream_stage(config_or_stage)


def _get_algo_cfg(config):
    return getattr(config, "algo", config)


def _get_train_cfg(config):
    return getattr(config, "train", config)


def get_algo_name(config_or_algo) -> str:
    if isinstance(config_or_algo, str):
        algo_name = config_or_algo
    else:
        algo_cfg = _get_algo_cfg(config_or_algo)
        if isinstance(algo_cfg, str):
            algo_name = algo_cfg
        else:
            algo_name = getattr(algo_cfg, "algo", None)
    if not isinstance(algo_name, str) or not algo_name:
        raise ValueError(f"Unsupported algorithm name: {algo_name!r}")
    return algo_name


def is_cascade_algo(config_or_algo) -> bool:
    return get_algo_name(config_or_algo).endswith(CASCADE_ALGO_SUFFIX)


def get_base_algo_name(config_or_algo) -> str:
    algo_name = get_algo_name(config_or_algo)
    if algo_name.endswith(CASCADE_ALGO_SUFFIX):
        algo_name = algo_name[:-len(CASCADE_ALGO_SUFFIX)]
    if algo_name not in BASE_ALGO_NAMES:
        raise ValueError(f"Unsupported base algorithm name: {algo_name!r}")
    return algo_name


def get_dim_skill(config) -> int:
    algo_cfg = _get_algo_cfg(config)
    return int(getattr(algo_cfg, "dim_skill", 0))


def uses_skill_inputs(config) -> bool:
    return not is_zero_training_stage(config) and get_dim_skill(config) > 0


def effective_skill_dim(config) -> int:
    return get_dim_skill(config) if uses_skill_inputs(config) else 0


def should_build_traj_encoder(config) -> bool:
    return is_pretraining_stage(config)


def should_build_skill_dynamics(config) -> bool:
    return is_pretraining_stage(config) and get_base_algo_name(config) == "dads"


def should_build_pretrain_auxiliaries(config) -> bool:
    return is_pretraining_stage(config)


def should_use_kme(config) -> bool:
    algo_cfg = _get_algo_cfg(config)
    return is_pretraining_stage(config) and bool(getattr(algo_cfg, "use_kme", False))


def should_update_target_traj_encoder(config) -> bool:
    algo_cfg = _get_algo_cfg(config)
    return is_pretraining_stage(config) and bool(getattr(algo_cfg, "use_target_traj_encoder", False))


def requires_best_skill_search(config) -> bool:
    return is_finetune_stage(config) and uses_skill_inputs(config)


def validate_stage_config(config) -> None:
    stage = get_stage(config)
    algo_cfg = _get_algo_cfg(config)
    train_cfg = _get_train_cfg(config)
    dim_skill = get_dim_skill(config)

    if stage == PRETRAIN_STAGE:
        if dim_skill <= 0:
            raise ValueError("[pre_training] requires dim_skill > 0.")
        return

    if stage == FINETUNE_STAGE:
        if dim_skill <= 0:
            raise ValueError("[finetune] requires dim_skill > 0.")
        skill_policy_path = getattr(train_cfg, "skill_policy_path", "")
        if not skill_policy_path:
            raise ValueError("[finetune] requires a non-empty skill_policy_path.")
        return

    if dim_skill != 0:
        raise ValueError("[zero_training] requires dim_skill == 0.")
    if bool(getattr(algo_cfg, "discrete", False)):
        raise ValueError("[zero_training] does not allow discrete skill configuration.")
    if bool(getattr(algo_cfg, "use_hierarchical_skill", False)):
        raise ValueError("[zero_training] does not allow use_hierarchical_skill.")
    if bool(getattr(algo_cfg, "use_hierarchical_policy", False)):
        raise ValueError("[zero_training] does not allow use_hierarchical_policy.")
    if bool(getattr(algo_cfg, "use_hierarchical_phi", False)):
        raise ValueError("[zero_training] does not allow use_hierarchical_phi.")
    if int(getattr(algo_cfg, "num_skill_levels", 1)) != 1:
        raise ValueError("[zero_training] requires num_skill_levels == 1.")
