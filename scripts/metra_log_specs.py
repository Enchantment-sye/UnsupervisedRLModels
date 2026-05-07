"""TensorBoard tag specs for METRA reproduction audits.

The audit script uses this module as the single source of truth for required
and optional TensorBoard tags. Tags are full TensorBoard scalar/image names.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_ENVS = (
    "kitchen",
    "ant",
    "dmc_quadruped",
    "dmc_humanoid",
    "dmc_cheetah",
    "half_cheetah",
)

TRAIN_METRIC_NAMES = frozenset(
    {
        "PureRewardMean",
        "PureRewardStd",
        "PureRewardMin",
        "PureRewardMax",
        "ScaledRewardMean",
        "ScaledRewardStd",
        "DeltaPhiNormMean",
        "DeltaPhiNormStd",
        "DeltaPhiNormMax",
        "Q1Mean",
        "Q2Mean",
        "QTargetsMean",
        "QTargetsStd",
        "QTdErrAbsMean",
        "LossSacp",
        "SacpNewActionLogProbMean",
        "Alpha",
        "LogAlpha",
        "AlphaLr",
        "LossAlpha",
        "DualLam",
        "LossDualLam",
        "DualCstPenalty",
        "TemporalViolationMean",
        "TemporalViolationFrac",
        "TotalGradNormAll",
        "TotalGradNormTrajEncoder",
        "TotalGradNormOptionPolicy",
        "TotalGradNormQf",
        "TotalGradNormDualLam",
        "TotalGradNormLogAlpha",
    }
)

TRAIN_TAGS = frozenset(f"TrainSp/METRA/{tag}" for tag in TRAIN_METRIC_NAMES)

EVAL_COMMON_TAGS = frozenset(
    {
        "eval/EvalOp/AverageReturn",
        "eval/EvalOp/AverageDiscountedReturn",
        "eval/EvalOp/ReturnOverall",
        "eval/EvalOp/NumTrajs",
    }
)

KITCHEN_TAGS = frozenset(
    {
        "eval/KitchenTaskBottomBurner",
        "eval/KitchenTaskLightSwitch",
        "eval/KitchenTaskSlideCabinet",
        "eval/KitchenTaskHingeCabinet",
        "eval/KitchenTaskMicrowave",
        "eval/KitchenTaskKettle",
        "eval/KitchenOverall",
        "eval/KitchenPolicyTaskCoverage",
        "eval/KitchenBottomBurnerSuccessRate",
        "eval/KitchenLightSwitchSuccessRate",
        "eval/KitchenSlideCabinetSuccessRate",
        "eval/KitchenHingeCabinetSuccessRate",
        "eval/KitchenMicrowaveSuccessRate",
        "eval/KitchenKettleSuccessRate",
        "eval/KitchenAvgCompletedTasksPerTraj",
        "eval/KitchenBestCompletedTasksPerTraj",
    }
)

LOCOMOTION_TAGS = frozenset(
    {
        "eval/MjNumTrajs",
        "eval/MjAvgTrajLen",
        "eval/MjNumCoords",
        "eval/MjNumUniqueCoords",
        "eval/PolicyStateCoverageXYBins",
        "eval/QueueStateCoverageXYBins",
        "eval/TotalStateCoverageXYBins",
        "eval/PolicyFinalXYDispMean",
        "eval/PolicyFinalXYDispMax",
        "eval/PolicyXRange",
        "eval/PolicyYRange",
        "eval/PolicyMeanSpeed",
    }
)

COMMON_OPTIONAL_TAGS = frozenset(
    {
        "TotalEnvSteps",
        "TimeTotal",
        "TotalEpoch",
        "TrainSp/METRA/AverageExternalDiscountedReturn",
        "TrainSp/METRA/AverageExternalReturn",
        "TrainSp/METRA/TimeTraining",
        "eval/MissingCoverageInfo",
    }
)

IMAGE_OPTIONAL_TAGS = frozenset(
    {
        "TrajPlot",
        "PhiPlot",
        "eval/TrajPlot",
        "eval/PhiPlot",
    }
)

LEGACY_ALIAS_TAGS = {
    "avg_completed_tasks": "eval/KitchenAvgCompletedTasksPerTraj",
    "eval/avg_completed_tasks": "eval/KitchenAvgCompletedTasksPerTraj",
    "eval/undiscounted_return": "eval/EvalOp/AverageReturn",
    "eval/TotalEnvSteps": "TotalEnvSteps",
    "eval/TotalEpoch": "TotalEpoch",
    "eval/TimeTotal": "TimeTotal",
    "eval/kitchen/policy_task_coverage": "eval/KitchenPolicyTaskCoverage",
    "eval/kitchen/overall_6task_coverage": "eval/KitchenOverall",
    "eval/kitchen/best_completed_tasks": "eval/KitchenBestCompletedTasksPerTraj",
}
LEGACY_ALIAS_TAGS.update(
    {f"METRA/{tag}": f"TrainSp/METRA/{tag}" for tag in TRAIN_METRIC_NAMES}
)


@dataclass(frozen=True)
class TagSpec:
    required_tags: frozenset[str]
    optional_tags: frozenset[str]


def get_tag_spec(env: str) -> TagSpec:
    """Return required and optional TensorBoard tags for a supported env."""

    if env not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env {env!r}; expected one of {SUPPORTED_ENVS}")

    required = set(TRAIN_TAGS) | set(EVAL_COMMON_TAGS)
    optional = set(COMMON_OPTIONAL_TAGS) | set(IMAGE_OPTIONAL_TAGS)

    if env == "kitchen":
        required.update(KITCHEN_TAGS)
        required.update(
            {
                "eval/KitchenQueueTaskCoverage",
                "eval/KitchenTotalTaskCoverage",
            }
        )
        optional.update(
            {
                "eval/KitchenMissingSuccessKeys",
                "eval/kitchen/bottom_burner_success",
                "eval/kitchen/light_switch_success",
                "eval/kitchen/slide_cabinet_success",
                "eval/kitchen/hinge_cabinet_success",
                "eval/kitchen/microwave_success",
                "eval/kitchen/kettle_success",
            }
        )
    else:
        required.update(LOCOMOTION_TAGS)

    return TagSpec(frozenset(required), frozenset(optional))
