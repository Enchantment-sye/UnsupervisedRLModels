from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .constraints import ConstraintContext, evaluate_numpy, project_to_box, stop_action_physical, torch_margins
from .types import SafetyReport


class LBSGDGovernor:
    def __init__(self, cfg, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def project(
            self,
            raw_action_physical: np.ndarray,
            ctx: ConstraintContext,
            policy_obs=None,
            *,
            runtime_lbsgd_steps: Optional[int] = None,
            runtime_barrier_eta: Optional[float] = None,
    ) -> Tuple[np.ndarray, SafetyReport]:
        report = SafetyReport(safety_enabled=True, safety_mode=str(getattr(self.cfg, "mode", "sim")))
        raw = np.asarray(raw_action_physical, dtype=np.float32).reshape(-1)[:ctx.action_dim]
        raw_eval = evaluate_numpy(raw, ctx, include_proxy=False)
        report.safety_raw_action_violation = raw_eval.violation_count > 0
        report.safety_min_margin = raw_eval.min_margin

        current = raw.copy()
        if raw_eval.min_margin <= ctx.min_barrier_margin:
            current, _, infeasible = project_to_box(raw, ctx)
            current_eval = evaluate_numpy(current, ctx, include_proxy=False)
            if infeasible or current_eval.min_margin <= ctx.min_barrier_margin:
                current, interior_found = self._find_feasible_interior(raw, current, ctx)
                current_eval = evaluate_numpy(current, ctx, include_proxy=False)
                if (not interior_found) or current_eval.min_margin <= ctx.min_barrier_margin:
                    report.safety_lbsgd_infeasible = True
                    report.with_infeasible("lbsgd_no_feasible_interior", source="lbsgd")
                    report.safety_min_margin = current_eval.min_margin
                    return current, report

        steps = int(runtime_lbsgd_steps if runtime_lbsgd_steps is not None else getattr(self.cfg, "lbsgd_steps", 0))
        if torch is None or steps <= 0:
            eval_current = evaluate_numpy(current, ctx, include_proxy=False)
            report.safety_min_margin = eval_current.min_margin
            return current.astype(np.float32), report

        eta = float(runtime_barrier_eta if runtime_barrier_eta is not None else getattr(self.cfg, "barrier_eta", 1e-2))
        lr = float(getattr(self.cfg, "lbsgd_lr", 1e-2))
        backtrack_steps = int(getattr(self.cfg, "lbsgd_backtrack_steps", 10))
        deviation_weight = float(getattr(self.cfg, "deviation_weight", 1.0))
        min_margin = float(getattr(self.cfg, "min_barrier_margin", ctx.min_barrier_margin))

        device = torch.device("cpu")
        raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(current, dtype=torch.float32, device=device)

        for _ in range(steps):
            action_t = action_t.detach().clone().requires_grad_(True)
            margins = torch_margins(action_t, ctx)
            if torch.any(margins <= min_margin):
                report.safety_lbsgd_infeasible = True
                report.with_infeasible("lbsgd_left_interior", source="lbsgd")
                break
            objective = 0.5 * deviation_weight * torch.sum((action_t - raw_t) ** 2)
            objective = objective - eta * torch.sum(torch.log(margins))
            objective.backward()
            grad = action_t.grad.detach()
            accepted = None
            for bt in range(backtrack_steps + 1):
                step_lr = lr * (0.5 ** bt)
                candidate = (action_t.detach() - step_lr * grad).cpu().numpy().astype(np.float32)
                candidate, _, infeasible = project_to_box(candidate, ctx)
                candidate_eval = evaluate_numpy(candidate, ctx, include_proxy=False)
                if not infeasible and candidate_eval.min_margin > min_margin:
                    accepted = candidate
                    report.safety_min_margin = min(report.safety_min_margin, candidate_eval.min_margin)
                    break
            if accepted is None:
                report.safety_lbsgd_infeasible = True
                report.with_infeasible("lbsgd_backtracking_failed", source="lbsgd")
                break
            action_t = torch.as_tensor(accepted, dtype=torch.float32, device=device)
            report.safety_lbsgd_steps += 1

        final = action_t.detach().cpu().numpy().astype(np.float32)
        final_eval = evaluate_numpy(final, ctx, include_proxy=False)
        report.safety_min_margin = min(report.safety_min_margin, final_eval.min_margin)
        return final, report

    def _find_feasible_interior(self, raw: np.ndarray, projected: np.ndarray, ctx: ConstraintContext):
        min_margin = float(getattr(self.cfg, "min_barrier_margin", ctx.min_barrier_margin))
        reference = stop_action_physical(raw, ctx)
        ref_eval = evaluate_numpy(reference, ctx, include_proxy=False)
        if ref_eval.min_margin <= min_margin:
            return projected.astype(np.float32), False
        best = reference.astype(np.float32)
        for alpha in np.linspace(0.95, 0.0, num=20, dtype=np.float32):
            candidate = reference + float(alpha) * (projected - reference)
            candidate_eval = evaluate_numpy(candidate, ctx, include_proxy=False)
            if candidate_eval.min_margin > min_margin:
                best = candidate.astype(np.float32)
                break
        return best.astype(np.float32), True
