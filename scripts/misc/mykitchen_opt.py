import cv2
import os
os.environ["MUJOCO_GL"] = "egl"
import akro
import gym
import d4rl
import numpy as np
from envs.wrappers import EnvSpec, Box
from envs.kitchen.kitchen import KitchenEnv

class MyKitchenEnv(KitchenEnv):

    def __init__(
            self,*args, **kwargs):
        self.use_pixel = kwargs.pop('use_pixel', True)
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.ob_info = dict(
            type='pixel',
            pixel_shape=(self._width, self._width, 3),
        )

    @property
    def obs_space(self):
        spaces = {
            "image": akro.Box(low=-np.inf, high=np.inf, shape=(self._width, self._width, 3)),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),

        }
        return spaces

    @property
    def act_space(self):
        asp = self._env.action_space
        return {"action": Box(asp.low, asp.high, asp.shape, asp.dtype)}

    @property
    def spec(self):
        return EnvSpec(obs_space=self.obs_space, act_space=self.act_space)

    def _get_obs(self, state):
        if self.use_pixel:
            image = self._env.render('rgb_array', width=self._env.imwidth, height=self._env.imheight)
        else:
            image = np.zeros((self._env.imwidth, self._env.imheight, 3), dtype=np.uint8)
            
        obs = {'image': image, 'state': state}
        
        if self.log_per_goal:
            for i, goal_idx in enumerate(self.goals):
                task_rel_success, all_obj_success = self.compute_success(goal_idx)
                obs['metric_success_task_relevant/goal_'+str(goal_idx)] = task_rel_success
                obs['metric_success_all_objects/goal_'+str(goal_idx)]   = all_obj_success
        if self.use_goal_idx:
            task_rel_success, all_obj_success = self.compute_success(self.goal_idx)
            obs['metric_success_task_relevant/goal_'+str(self.goal_idx)] = task_rel_success
            obs['metric_success_all_objects/goal_'+str(self.goal_idx)]   = all_obj_success

        return obs

    def get_state(self, state):
        image = state['image']
        return image.flatten()

    def reset(self):
        state = super().reset()
        ob = self.get_state(state)
        self.last_state = state
        self.last_ob = ob
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": ob,
            "success": False,
            "info": dict(),
        }
        return obs

    def step(self, action, render=False):
        next_state, reward, done, info = super().step(action)
        ob = self.get_state(next_state)

        coords = self.last_state['state'][:2].copy()
        next_coords = next_state['state'][:2].copy()
        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_state['state']
        info['next_ori_obs'] = next_state['state']
        if render:
            info['render'] = next_state['image'].transpose(2, 0, 1)

        self.last_state = next_state
        self.last_ob = ob
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": ob,
            "success": done,
            "info": dict(),
        }
        return obs
