import numpy as np
import torch
import torch.nn as nn

from utils import utils
from models.layers import MultiHeadedMLPModule
from models.distributions import GaussianMLPTwoHeadedModuleEx

from core.actor_critic_factory import ActorCriticModuleFactory

class CascadeStageFactory:
    """
    Picklable factory class for creating Cascade stages.
    Replaces the local closure 'stage_factory_fn'.
    """
    def __init__(self, 
                 obs_dim,
                 skill_dim,
                 skill_input_dim,
                 num_skill_levels,
                 use_hierarchical_policy,
                 use_hierarchical_skill,
                 action_dim, 
                 hidden_sizes, 
                 nonlinearity, 
                 actor_init_std, 
                 actor_max_log_std,
                 simba_hp=None,
                 ac_backbone='mlp'):
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.skill_input_dim = skill_input_dim
        self.num_skill_levels = num_skill_levels
        self.use_hierarchical_policy = use_hierarchical_policy
        self.use_hierarchical_skill = use_hierarchical_skill
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.actor_init_std = actor_init_std
        self.actor_max_log_std = actor_max_log_std
        self.simba_hp = simba_hp
        self.ac_backbone = ac_backbone

    def _input_dim_for_stage(self, stage_number):
        if not self.use_hierarchical_policy:
            return self.obs_dim + self.skill_input_dim
        if not self.use_hierarchical_skill:
            return self.obs_dim + self.skill_dim
        return self.obs_dim + min(stage_number, self.num_skill_levels) * self.skill_dim

    def __call__(self, stage_number, is_cascade_stage=False, gate_type='scalar'):
        input_dim = self._input_dim_for_stage(stage_number)
        if is_cascade_stage:
            return CascadeStage(
                input_dim=input_dim,
                output_dim=self.action_dim,
                hidden_sizes=self.hidden_sizes,
                gate_type=gate_type,
                hidden_nonlinearity=self.nonlinearity,
                output_w_init=torch.nn.init.xavier_normal_,
                init_std=float(self.actor_init_std),
                max_std=np.exp(float(self.actor_max_log_std)),
                normal_distribution_cls=utils.TanhNormal,
            )
        else:
            return ActorCriticModuleFactory.create_actor_core(
                backbone=self.ac_backbone,
                input_dim=input_dim,
                action_dim=self.action_dim,
                mlp_hidden_sizes=self.hidden_sizes,
                mlp_nonlinearity=self.nonlinearity,
                actor_init_std=self.actor_init_std,
                actor_max_log_std=self.actor_max_log_std,
                simba_hp=self.simba_hp,
            )

class CascadeStage(GaussianMLPTwoHeadedModuleEx):
    """
    A single stage in the Cascade Actor for k >= 1.
    Outputs: mean, log_std, lambda_logit.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(256, 256),
                 gate_type='scalar',
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=utils.TanhNormal,
                 **kwargs):
        # We initialize super with dummy values because we will overwrite the network
        # This is to satisfy the inheritance structure and reuse methods
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         learn_std=learn_std,
                         init_std=init_std,
                         min_std=min_std,
                         max_std=max_std,
                         std_parameterization=std_parameterization,
                         layer_normalization=layer_normalization,
                         normal_distribution_cls=normal_distribution_cls)
        
        self.gate_type = gate_type
        
        # Heads: [mean, log_std, lambda_logit]
        output_dims = [output_dim, output_dim]
        if gate_type == 'scalar':
            output_dims.append(1)
        else:
            output_dims.append(output_dim)
            
        # Overwrite the network created by super().__init__
        self._shared_mean_log_std_network = MultiHeadedMLPModule(
            n_heads=3,
            input_dim=self._input_dim,
            output_dims=output_dims,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_, # mean
                # log_std
                (lambda x: nn.init.constant_(x, self._init_std.item())
                 if self._std_parameterization not in ['softplus_real']
                 else lambda x: nn.init.constant_(x, self._init_std.exp().exp().add(-1.0).log().item())),
                # lambda_logit
                nn.init.zeros_ 
            ],
            layer_normalization=self._layer_normalization,
            **kwargs
        )

    def _get_mean_and_log_std(self, *inputs):
        # Return first two heads
        outs = self._shared_mean_log_std_network(*inputs)
        return outs[0], outs[1]

    def forward_stage(self, *inputs):
        """Returns mean, log_std, lambda_logit"""
        outs = self._shared_mean_log_std_network(*inputs)
        return outs[0], outs[1], outs[2]


class CascadeActor(nn.Module):
    """
    Cascade Actor composed of multiple stages.
    Stage 0: Standard Gaussian Policy (GaussianMLPTwoHeadedModuleEx)
    Stage k>0: CascadeStage (outputs mean, std, lambda)
    """
    def __init__(self, 
                 stage_factory_fn, 
                 device,
                 gate_type='scalar',
                 min_lambda=0.01,
                 max_lambda=0.99,
                 use_hierarchical_policy=False,
                 use_hierarchical_skill=False,
                 num_skill_levels=1,
                 dim_skill=0,
                 obs_dim=None,
                 skill_input_dim=None):
        super().__init__()
        self.stage_factory_fn = stage_factory_fn
        self.device = device
        self.gate_type = gate_type
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.use_hierarchical_policy = use_hierarchical_policy
        self.use_hierarchical_skill = use_hierarchical_skill
        self.num_skill_levels = num_skill_levels
        self.dim_skill = dim_skill
        self.obs_dim = obs_dim
        self.skill_input_dim = skill_input_dim if skill_input_dim is not None else dim_skill

        self.stages = nn.ModuleList()
        self.add_stage(is_first=True)
        
    def add_stage(self, is_first=False, init_from_prev=True):
        stage_number = len(self.stages) + 1
        if is_first:
            stage = self.stage_factory_fn(stage_number=stage_number, is_cascade_stage=False)
        else:
            stage = self.stage_factory_fn(stage_number=stage_number, is_cascade_stage=True, gate_type=self.gate_type)
            
            if init_from_prev and len(self.stages) > 0:
                prev_stage = self.stages[-1]
                self._init_stage_from_prev(stage, prev_stage)
                
        stage = stage.to(self.device)
        self.stages.append(stage)
        return stage
        
    def _init_stage_from_prev(self, new_stage, prev_stage):
        # Helper to copy weights where possible
        src_net = getattr(prev_stage, '_shared_mean_log_std_network', None)
        dst_net = getattr(new_stage, '_shared_mean_log_std_network', None)
        
        if src_net is None or dst_net is None:
            return

        if hasattr(src_net, '_layers') and hasattr(dst_net, '_layers'):
            for src_seq, dst_seq in zip(src_net._layers, dst_net._layers):
                if len(src_seq) > 0 and len(dst_seq) > 0:
                    src_linear = src_seq[0]
                    dst_linear = dst_seq[0]
                    if isinstance(src_linear, nn.Linear) and isinstance(dst_linear, nn.Linear):
                        rows = min(dst_linear.weight.data.shape[0], src_linear.weight.data.shape[0])
                        cols = min(dst_linear.weight.data.shape[1], src_linear.weight.data.shape[1])
                        dst_linear.weight.data[:rows, :cols].copy_(src_linear.weight.data[:rows, :cols])
                        if dst_linear.bias is not None and src_linear.bias is not None:
                            bias_dim = min(dst_linear.bias.data.shape[0], src_linear.bias.data.shape[0])
                            dst_linear.bias.data[:bias_dim].copy_(src_linear.bias.data[:bias_dim])
        
        if hasattr(src_net, '_output_layers') and hasattr(dst_net, '_output_layers'):
            for i in range(2):
                if i < len(src_net._output_layers) and i < len(dst_net._output_layers):
                    src_seq = src_net._output_layers[i]
                    dst_seq = dst_net._output_layers[i]
                    if len(src_seq) > 0 and len(dst_seq) > 0:
                        src_linear = src_seq[0]
                        dst_linear = dst_seq[0]
                        if isinstance(src_linear, nn.Linear) and isinstance(dst_linear, nn.Linear):
                            rows = min(dst_linear.weight.data.shape[0], src_linear.weight.data.shape[0])
                            cols = min(dst_linear.weight.data.shape[1], src_linear.weight.data.shape[1])
                            dst_linear.weight.data[:rows, :cols].copy_(src_linear.weight.data[:rows, :cols])
                            if dst_linear.bias is not None and src_linear.bias is not None:
                                bias_dim = min(dst_linear.bias.data.shape[0], src_linear.bias.data.shape[0])
                                dst_linear.bias.data[:bias_dim].copy_(src_linear.bias.data[:bias_dim])

    def _split_observations(self, observations):
        if self.skill_input_dim == 0:
            return observations, None

        obs = observations[..., :self.obs_dim]
        skill = observations[..., self.obs_dim:self.obs_dim + self.skill_input_dim]
        if self.use_hierarchical_skill:
            skill = skill.reshape(skill.shape[0], self.num_skill_levels, self.dim_skill)
        return obs, skill

    def _stage_observation(self, observations, stage_index):
        if not self.use_hierarchical_policy:
            return observations

        obs, skill = self._split_observations(observations)
        if skill is None:
            return obs
        if not self.use_hierarchical_skill:
            return torch.cat([obs, skill], dim=-1)

        active_levels = min(stage_index + 1, self.num_skill_levels)
        skill_prefix = skill[:, :active_levels, :].reshape(skill.shape[0], -1)
        return torch.cat([obs, skill_prefix], dim=-1)
            
    def forward(self, observations):
        stage0 = self.stages[0]
        stage_obs = self._stage_observation(observations, 0)
        mean_prev, log_std_prev_uncentered = stage0._get_mean_and_log_std(stage_obs)
        std_prev = self._compute_std(stage0, log_std_prev_uncentered)
        var_prev = std_prev.pow(2)
        
        for i in range(1, len(self.stages)):
            stage = self.stages[i]
            stage_obs = self._stage_observation(observations, i)
            mean_k, log_std_k_uncentered, lam_logit_k = stage.forward_stage(stage_obs)
            
            std_k = self._compute_std(stage, log_std_k_uncentered)
            var_k = std_k.pow(2)
            
            lam_k = torch.sigmoid(lam_logit_k)
            lam_k = self.min_lambda + (self.max_lambda - self.min_lambda) * lam_k
            
            mean_prev = lam_k * mean_prev + (1 - lam_k) * mean_k
            var_prev = (lam_k.pow(2)) * var_prev + ((1 - lam_k).pow(2)) * var_k
            
        std_ens = var_prev.sqrt()
        
        # Construct TanhNormal
        dist = utils.TanhNormal(mean_prev, std_ens)
        return dist

    def _compute_std(self, module, log_std_uncentered):
        # Replicates GaussianMLPBaseModule logic for std
        if module._min_std_param is not None or module._max_std_param is not None:
            min_val = None if module._min_std_param is None else module._min_std_param.item()
            max_val = None if module._max_std_param is None else module._max_std_param.item()
            log_std_uncentered = log_std_uncentered.clamp(min=min_val, max=max_val)

        if module._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        elif module._std_parameterization == 'softplus':
            std = log_std_uncentered.exp().exp().add(1.).log()
        elif module._std_parameterization == 'softplus_real':
            std = log_std_uncentered.exp().add(1.).log()
        else:
            raise NotImplementedError
        return std

    def forward_mode(self, observations):
        dist = self.forward(observations)
        return dist.mean
