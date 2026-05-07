import cv2
import os
os.environ["MUJOCO_GL"] = "egl"
import akro
import gym
import d4rl  # 确保注册
import numpy as np
from envs.wrappers import EnvSpec, Box
from envs.kitchen.kitchen import KitchenEnv
from envs.kitchen.metrics import calc_kitchen_eval_metrics

class MyKitchenEnv(KitchenEnv):

    def __init__(
            self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.ob_info = dict(
            type='pixel',
            pixel_shape=(self._width, self._width, 3),
        )

    # -------------------- 空间定义 --------------------
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
        # return akro.Box(low=-np.inf, high=np.inf, shape=(256, 256, 3))


    @property
    def act_space(self):
        asp = self._env.action_space
        return {"action": Box(asp.low, asp.high, asp.shape, asp.dtype)}

    @property
    def spec(self):
        return EnvSpec(obs_space=self.obs_space, act_space=self.act_space)

    # -------------------- 交互接口 --------------------
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
            "state": state["state"],
            "info": {
                "state": state["state"],
            },
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
        info['state'] = next_state['state']
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
            # "image": self._env.sim.render(
            #     *self._size, mode="offscreen", camera_name=self._camera
            # ),
            "state": self.last_state['state'],
            "success": done,
            "info": info,
        }
        return obs

    def calc_eval_metrics(self, trajectories, is_option_trajectories=True, coord_dims=None):
        return calc_kitchen_eval_metrics(trajectories)

