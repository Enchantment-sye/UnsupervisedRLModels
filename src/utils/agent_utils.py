import numpy as np
import torch
from collections import defaultdict
from utils import utils

def extract_state_from_obs(obs):
    if isinstance(obs, dict):
        for key in ("state", "obs", "observation"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        candidate = None
        for k, v in obs.items():
            if k == "image": continue
            arr = np.asarray(v)
            if arr.ndim == 1:
                candidate = arr.astype(np.float32)
                break
        if candidate is None:
            raise RuntimeError(f"Cannot extract state from obs keys: {list(obs.keys())}")
        return candidate
    return np.asarray(obs, dtype=np.float32)

def flatten_data(data, device):
    epoch_data = {}
    for key, value in data.items():
        epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=device)
    return epoch_data


def numpy_batch_to_torch(value, device, *, dtype=torch.float32, non_blocking=False):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype, non_blocking=non_blocking)

    tensor = torch.from_numpy(np.asarray(value))
    if non_blocking and torch.device(device).type == "cuda":
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

def process_samples(paths, discount):
    data = defaultdict(list)
    agent_optional_fields = {
        'raw_action': 'raw_actions',
        'safe_action': 'safe_actions',
        'safety_correction_norm': 'safety_correction_norm',
    }
    env_optional_fields = {
        'safety_min_margin': 'safety_min_margin',
        'safety_redline_count': 'safety_redline_count',
        'safety_infeasible': 'safety_infeasible',
        'safety_qp_infeasible': 'safety_qp_infeasible',
        'safety_lbsgd_infeasible': 'safety_lbsgd_infeasible',
        'safety_qp_active': 'safety_qp_active',
        'safety_raw_action_violation': 'safety_raw_action_violation',
        'safety_safe_action_violation': 'safety_safe_action_violation',
    }
    for path in paths:
        data['obs'].append(path['observations'])
        data['next_obs'].append(path['next_observations'])
        data['actions'].append(path['actions'])
        data['rewards'].append(path['rewards'])
        data['dones'].append(path['dones'])
        data['returns'].append(utils.discount_cumsum(path['rewards'], discount))
        
        if 'pre_tanh_value' in path['agent_infos']:
            data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
        if 'log_prob' in path['agent_infos']:
            data['log_probs'].append(path['agent_infos']['log_prob'])
        if 'skill' in path['agent_infos']:
            data['skills'].append(path['agent_infos']['skill'])
            # Shift skills for next_skills? Assuming skill is constant over traj usually, but here we construct it.
            # Logic from metra_new.py:
            data['next_skills'].append(np.concatenate([path['agent_infos']['skill'][1:], path['agent_infos']['skill'][-1:]], axis=0))

        agent_infos = path.get('agent_infos', {}) or {}
        for src_key, dst_key in agent_optional_fields.items():
            if src_key in agent_infos:
                data[dst_key].append(np.asarray(agent_infos[src_key]))

        env_infos = path.get('env_infos', {}) or {}
        for src_key, dst_key in env_optional_fields.items():
            if src_key in env_infos:
                data[dst_key].append(np.asarray(env_infos[src_key]))

    return data

def get_mini_tensors(epoch_data, batch_size):
    # On-policy sampling
    idxs = np.random.choice(len(epoch_data['actions']), batch_size)
    data = {k: v[idxs] for k, v in epoch_data.items()}
    return data
