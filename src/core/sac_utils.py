import torch
from torch.nn import functional as F


def _optimizer_lr(algo, optimizer_key):
    optimizer = getattr(getattr(algo, "optimizer", None), "_optimizers", {}).get(optimizer_key)
    if optimizer is None or not optimizer.param_groups:
        return None
    return optimizer.param_groups[0].get("lr", None)


def _extract_action_dist(policy_output):
    if isinstance(policy_output, tuple):
        return policy_output[0]
    return policy_output


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)

    return actions + clip


def update_loss_qf(algo, metrics, v, obs, actions, next_obs, dones, rewards, policy, 
                   qf1=None, qf2=None, target_qf1=None, target_qf2=None):
    """Critic (Q-function) loss for SAC.

    Notes:
      - All *target* computations (policy(next_obs), target Qs, log_probs, etc.) run under
        torch.no_grad() to avoid unnecessary graph creation / VRAM use.
      - We temporarily switch the policy to eval() so BN/Dropout do not update running
        states during critic-only updates (e.g., warmup).
    """
    qf1 = qf1 or algo.qf1
    qf2 = qf2 or algo.qf2
    target_qf1 = target_qf1 or algo.target_qf1
    target_qf2 = target_qf2 or algo.target_qf2

    q1_pred = qf1(obs, actions).flatten()
    q2_pred = qf2(obs, actions).flatten()

    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

        was_training = policy.training
        policy.eval()
        try:
            next_action_dists = _extract_action_dist(policy(next_obs))
            if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
                pre_tanh, next_actions = next_action_dists.rsample_with_pre_tanh_value()
                next_action_log_probs = next_action_dists.log_prob(
                    next_actions, pre_tanh_value=pre_tanh
                ).flatten()
            else:
                next_actions = next_action_dists.rsample()
                next_action_log_probs = next_action_dists.log_prob(next_actions).flatten()

            if algo._env_spec is not None:
                next_actions = _clip_actions(algo, next_actions)

            target_q_values = torch.min(
                target_qf1(next_obs, next_actions).flatten(),
                target_qf2(next_obs, next_actions).flatten(),
            )

            target_q_values = target_q_values - alpha * next_action_log_probs
            target_q_values = algo.discount * target_q_values
            q_target = rewards + target_q_values * (1.0 - dones)
        finally:
            policy.train(was_training)

    loss_qf1 = F.mse_loss(q1_pred, q_target)
    loss_qf2 = F.mse_loss(q2_pred, q_target)
    td_err_abs = 0.5 * ((q1_pred - q_target).abs().mean() + (q2_pred - q_target).abs().mean())

    metrics.update({
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
        'Q1Mean': q1_pred.detach().mean(),
        'Q2Mean': q2_pred.detach().mean(),
        'QTargetsMean': q_target.detach().mean(),
        'QTargetsStd': q_target.detach().std(unbiased=False),
        'QTdErrAbsMean': td_err_abs.detach(),
        'ScaledRewardMean': rewards.detach().mean(),
        'ScaledRewardStd': rewards.detach().std(unbiased=False),
    })


def update_loss_sacp(
        algo, metrics, v,
        obs,
        policy,
        qf1=None, qf2=None
):
    qf1 = qf1 or algo.qf1
    qf2 = qf2 or algo.qf2

    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    action_dists = _extract_action_dist(policy(obs))
    if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
    else:
        new_actions = action_dists.rsample()
        if algo._env_spec is not None:
            new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)

    min_q_values = torch.min(
        qf1(obs, new_actions).flatten(),
        qf2(obs, new_actions).flatten(),
    )

    loss_sacp_base = (alpha * new_action_log_probs - min_q_values).mean()
    loss_sacp = loss_sacp_base
    distill_weight = float(getattr(algo, "safe_action_distill_weight", 0.0))
    if distill_weight > 0.0 and "safe_actions" in v:
        safe_actions = v["safe_actions"].to(device=new_actions.device, dtype=new_actions.dtype)
        if safe_actions.dim() > 2:
            safe_actions = safe_actions.reshape(safe_actions.shape[0], -1)
        if safe_actions.shape == new_actions.shape:
            loss_distill = F.mse_loss(new_actions, safe_actions)
            loss_sacp = loss_sacp + distill_weight * loss_distill
            metrics["LossSafeActionDistill"] = loss_distill
        else:
            metrics["LossSafeActionDistillSkipped"] = torch.tensor(1.0, device=new_actions.device)

    metrics.update({
        'SacpNewActionLogProbMean': new_action_log_probs.detach().mean(),
        'LossSacpBase': loss_sacp_base.detach(),
        'LossSacp': loss_sacp,
    })

    v.update({
        'new_action_log_probs': new_action_log_probs,
        'new_actions': new_actions,
    })


def update_loss_alpha(
        algo, metrics, v,
):
    loss_alpha = (-algo.log_alpha.param * (
            v['new_action_log_probs'].detach() + algo._target_entropy
    )).mean()
    alpha_lr = _optimizer_lr(algo, 'log_alpha')
    if alpha_lr is None:
        alpha_lr = 0.0

    metrics.update({
        'Alpha': algo.log_alpha.param.exp().detach(),
        'LogAlpha': algo.log_alpha.param.detach(),
        'AlphaLr': torch.as_tensor(alpha_lr, device=algo.log_alpha.param.device, dtype=algo.log_alpha.param.dtype),
        'LossAlpha': loss_alpha,
    })


def update_targets(algo, pairs=None):
    """Update parameters in the target q-functions.
    
    Args:
        algo: The algorithm instance (containing tau).
        pairs: Optional list of (target_module, source_module) tuples. 
               If None, defaults to algo.target_qf1/2 and algo.qf1/2.
    """
    if pairs is None:
        pairs = [
            (algo.target_qf1, algo.qf1),
            (algo.target_qf2, algo.qf2)
        ]
        
    for target_module, source_module in pairs:
        for t_param, param in zip(target_module.parameters(), source_module.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) +
                               param.data * algo.tau)
