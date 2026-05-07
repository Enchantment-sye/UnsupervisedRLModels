import numpy as np
import torch
import logging
from tqdm import tqdm

from utils import utils


class SkillSelector:
    def __init__(self, env, actor, worker, device, logger=None):
        self.env = env
        self.actor = actor
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        # Reuse agent's rollout worker for faster env interaction
        self.worker =worker

    def select(self):
        raise NotImplementedError

    def evaluate_skill(self, skill, num_episodes):
        if self.worker is not None and self.actor is not None:
            total_ret = 0.0
            for _ in range(num_episodes):
                batch = self.worker.rollout(
                    self.env,
                    self.actor,
                    extra={"skill": skill},
                    deterministic_policy=False # Use stochastic policy for diverse trajectories
                )
                trajs = batch.to_trajectory_list()
                # Sum rewards of the episode
                total_ret += float(sum([float(tr["rewards"].sum()) for tr in trajs]))
            return total_ret / num_episodes
        # Fallback: step loop
        avg_ret = 0.0
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            max_steps = getattr(self.worker, "_time_limit", None) or 1000
            while not done:
                prev = obs["image"] if isinstance(obs, dict) and ("image" in obs) else obs
                ll_in = utils.get_np_concat_obs(prev, skill)
                action, _ = self.actor.get_action(ll_in)
                ts = self.env.step({"action": action})
                ep_ret += float(ts.get("reward", 0.0))
                done = bool(ts.get("is_terminal", ts.get("is_last", False)))
                obs = ts
                steps += 1
                if steps >= max_steps:
                    break
            avg_ret += ep_ret
        return avg_ret / num_episodes

class DiscreteSkillSelector(SkillSelector):
    def __init__(self, env, actor, worker,  device, dim_skill, num_episodes=3, logger=None):
        super().__init__(env, actor, worker, device, logger)
        self.dim_skill = dim_skill
        self.num_episodes = num_episodes

    def select(self):
        self.logger.info("Selecting best discrete skill...")
        best_return = -float('inf')
        best_skill = None
        
        # Iterate over one-hot skills
        for i in tqdm(range(self.dim_skill), desc="Eval Discrete Skills"):
            skill = np.eye(self.dim_skill)[i]
            avg_ret = self.evaluate_skill(skill, self.num_episodes)
            
            if avg_ret > best_return:
                best_return = avg_ret
                best_skill = skill
        self.logger.info(f"Best Discrete Skill: index={np.argmax(best_skill)}, return={best_return}")
        print(f"Best Discrete Skill: index={np.argmax(best_skill)}, return={best_return}")
        return best_skill

class CEMSkillSelector(SkillSelector):
    def __init__(self, env, actor, worker, device, dim_skill,
                 cem_iters=10, 
                 cem_pop_size=16,
                 cem_elites=10, 
                 cem_alpha=0.1, 
                 update_mode='standard', 
                 cem_temperature=1.0,
                 cem_epsilon=1e-5,
                 num_episodes=1,
                 logger=None):
        super().__init__(env, actor, worker, device, logger)
        self.dim_skill = dim_skill
        self.cem_iters = cem_iters
        self.cem_pop_size = cem_pop_size
        self.cem_elites = int(cem_elites)
        self.cem_alpha = cem_alpha
        self.update_mode = update_mode
        self.cem_temperature = cem_temperature
        self.cem_epsilon = cem_epsilon
        self.num_episodes = num_episodes

    def select(self):
        self.logger.info(f"Selecting best continuous skill using CEM ({self.update_mode})...")
        
        # Initialize distribution N(0, I)
        mu = np.zeros(self.dim_skill)
        sigma = np.eye(self.dim_skill)
        
        best_overall_ret = -float('inf')
        best_overall_skill = None

        for itr in range(self.cem_iters):
            # 1. Sample candidates: z ~ N(mu, sigma)
            # Use multivariate_normal handles covariance matrix properly
            candidates = np.random.multivariate_normal(mu, sigma, size=self.cem_pop_size)
            
            # Normalize to unit sphere
            candidates = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
            
            # 2. Evaluate candidates
            returns = []
            for z in candidates:
                # Ensure z is float32
                ret = self.evaluate_skill(z.astype(np.float32), self.num_episodes)
                returns.append(ret)
            returns = np.array(returns)
            
            # Track best
            max_idx = np.argmax(returns)
            if returns[max_idx] > best_overall_ret:
                best_overall_ret = returns[max_idx]
                best_overall_skill = candidates[max_idx]
            
            # 3. Select Elites
            # Argsort returns indices from low to high, so take last 'elites'
            elite_idxs = np.argsort(returns)[-self.cem_elites:]
            elites = candidates[elite_idxs]
            elite_returns = returns[elite_idxs]
            
            self.logger.info(f"CEM Iter {itr}: Mean Ret={np.mean(returns):.10f}, Max Ret={np.max(returns):.10f}, Best Overall={best_overall_ret:.2f}")
            print(f"CEM Iter {itr}: Mean Ret={np.mean(returns):.10f}, Max Ret={np.max(returns):.10f}, Best Overall={best_overall_ret:.2f}")

            # 4. Update Parameters
            if self.update_mode == 'standard':
                new_mu = np.mean(elites, axis=0)
                # Covariance: (z-mu)(z-mu)^T
                # np.cov expects rowvar=False for (N, D)
                new_sigma = np.cov(elites, rowvar=False) + self.cem_epsilon * np.eye(self.dim_skill)
                
            elif self.update_mode == 'weighted':
                # Weights w_i propto exp(beta * J)
                weights = np.exp(self.cem_temperature * elite_returns)
                weights = weights / np.sum(weights)
                
                # Weighted Mean
                new_mu = np.sum(elites * weights[:, None], axis=0)
                
                # Weighted Covariance
                diff = elites - new_mu
                # Sigma = sum(w_i * diff_i * diff_i^T)
                # Efficient way: (weights * diff.T) @ diff
                new_sigma = (diff.T * weights) @ diff + self.cem_epsilon * np.eye(self.dim_skill)
                
            else: # Fallback to standard
                 new_mu = np.mean(elites, axis=0)
                 new_sigma = np.cov(elites, rowvar=False) + self.cem_epsilon * np.eye(self.dim_skill)
            
            # Smooth updates (Lazy update) if alpha provided
            # mu_new = (1-alpha)mu_old + alpha*mu_new
            if self.cem_alpha < 1.0:
                mu = (1 - self.cem_alpha) * mu + self.cem_alpha * new_mu
                sigma = (1 - self.cem_alpha) * sigma + self.cem_alpha * new_sigma
            else:
                mu = new_mu
                sigma = new_sigma

        self.logger.info(f"CEM Finished. Best Return: {best_overall_ret}")
        print(f"CEM Finished. Best Return: {best_overall_ret}")
        return best_overall_skill.astype(np.float32)
