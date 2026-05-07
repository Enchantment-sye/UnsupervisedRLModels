from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .constraints import ConstraintContext, evaluate_numpy, project_to_box
from .types import SafetyReport


class QPShield:
    def __init__(self, cfg, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def project(self, action_physical: np.ndarray, ctx: ConstraintContext) -> Tuple[np.ndarray, SafetyReport]:
        report = SafetyReport(safety_enabled=True, safety_mode=str(getattr(self.cfg, "mode", "sim")))
        projected, num_clipped, infeasible = project_to_box(action_physical, ctx)
        report.safety_qp_active = bool(num_clipped > 0 or infeasible)
        report.details["safety_qp_num_clipped"] = int(num_clipped)
        eval_after = evaluate_numpy(projected, ctx, include_proxy=True)
        report.safety_min_margin = eval_after.min_margin
        report.safety_safe_action_violation = eval_after.violation_count > 0
        if infeasible or eval_after.min_margin < -1e-7:
            report.safety_qp_infeasible = True
            report.with_infeasible("qp_infeasible", source="qp")
        return projected, report
