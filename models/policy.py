import torch
import torch.nn as nn
import numpy as np
from utils import utils

class PolicyEx(torch.nn.Module):
    def __init__(self,
                 name,
                 env_spec,
                 module,
                 clip_action=False,
                 omit_obs_idxs=None,
                 skill_info=None,
                 force_use_mode_actions=False,
                 ):
        super().__init__()

        self._name = name
        self.env_spec = env_spec
        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs

        self._skill_info = skill_info
        self._force_use_mode_actions = force_use_mode_actions

        self._module = module

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """
        return self._name

    @property
    def observation_space(self):
        """The observation space for the environment.

        Returns:
            akro.Space: Observation space.

        """
        return self.env_spec.observation_space

    @property
    def action_space(self):
        """The action space for the environment.

        Returns:
            akro.Space: Action space.

        """
        return self.env_spec.action_space

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self.state_dict()

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        self.load_state_dict(state_dict)

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """
        return


    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        observations = self.process_observations(observations)
        dist = self._module(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self._module.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self._module.forward_with_transform(observations, transform=transform)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (dict(mean=ret_mean, log_std=ret_log_std),
                    dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed))
        except NotImplementedError:
            info = (dict(),
                    dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self._module.forward_with_chunks(observations,
                                                merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            dist, info = self.forward(observations)
            if isinstance(dist, utils.TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(observations, torch.Tensor)
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(observations)
        else:
            actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
                )
        return actions, info

    def get_action(self, observation):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(next(self.parameters()).device)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}
