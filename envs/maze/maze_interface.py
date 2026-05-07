# ===== DMC 风格统一包装：LocoMazeEnv / MazeEnv / BallEnv =====
import os
import cv2
import gym
import numpy as np

from envs.maze.maze import make_maze_env


class LocoMazeEnv:
    """
    统一的 DMC 风格迷宫环境包装器。

    根据 domain 参数：
    - domain == "maze"：使用 make_maze_env(..., maze_env_type="maze")
    - domain == "ball"：使用 make_maze_env(..., maze_env_type="ball")

    对外统一接口：
    - obs_space / act_space
    - reset() -> dict
    - step({"action": ...}) -> dict
    - render() -> (H, W, 3) uint8
    """

    def __init__(
            self,
            task_type,       # "maze" 或 "ball"
            name: str,
            maze_unit: float = 4.0,
            maze_height: float = 0.5,
            terminate_at_goal: bool = True,
            success_timing: str = "post",
            add_noise_to_goal: bool = True,
            reward_task_id=None,
            use_oracle_rep: bool = False,
            action_repeat: int = 1,
            size=(64, 64),
            flatten_obs: bool = True,
            **kwargs,
    ):
        assert task_type in ("maze", "ball"), f"Unknown domain '{task_type}', expected 'maze' or 'ball'"

        os.environ.setdefault("MUJOCO_GL", "egl")

        domain, maze_type, ob_type = name.split("_", 2)

        # 调用你原来的工厂函数，创建底层“迷宫 + 机体”环境
        # 注意：强制 ob_type='pixels'，统一输出 image
        self._env = make_maze_env(
            loco_env_type=domain,
            maze_env_type=task_type,
            maze_type=maze_type,
            maze_unit=maze_unit,
            maze_height=maze_height,
            terminate_at_goal=terminate_at_goal,
            success_timing=success_timing,
            ob_type=ob_type,
            add_noise_to_goal=add_noise_to_goal,
            reward_task_id=reward_task_id,
            use_oracle_rep=use_oracle_rep,
            **kwargs,
        )

        self._action_repeat = int(action_repeat)
        self._size = tuple(size)
        self.flatten_obs = bool(flatten_obs)

        # 用 "states" 观测的形状定义 state 空间
        state_example = self._env.get_ob(ob_type="states")
        self._state_shape = state_example.shape

        # image / state 空间
        self._image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self._size + (3,),
            dtype=np.uint8,
        )
        self._state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._state_shape,
            dtype=np.float32,
        )

        # 动作空间：底层 env.action_space -> dict{"action": Box}
        act_space = self._env.action_space
        self._act_space = {
            "action": gym.spaces.Box(
                low=act_space.low.astype(np.float32),
                high=act_space.high.astype(np.float32),
                dtype=np.float32,
            )
        }


    @property
    def obs_space(self):
        return {
            "image": self._image_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "state": self._state_space,
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "info": dict(),  # 占位，与 DMC 保持相同结构
        }

    @property
    def act_space(self):
        return self._act_space

    # ----------------- 渲染 & 状态 -----------------

    def _get_state(self) -> np.ndarray:
        """使用底层 env 的 'states' 观测作为 state。"""
        state = self._env.get_ob(ob_type="states").astype(np.float32)
        return state

    def render(self, mode: str = "offscreen"):
        """
        取底层 env 的像素图像，resize 到 self._size，并可选展平。
        """
        img = self._env.render()  # H x W x 3, uint8
        # 若底层分辨率不等于目标 size，则缩放
        if (img.shape[0], img.shape[1]) != self._size:
            img = cv2.resize(
                img,
                (self._size[1], self._size[0]),  # cv2: (width, height)
                interpolation=cv2.INTER_AREA,
            )
        if self.flatten_obs:
            img = img.reshape(-1)
        return img

    # ----------------- reset / step -----------------

    def reset(self):
        """
        与 DMC.reset 类似：
        - 调用底层 env.reset()
        - is_first=True，其它 flag=False
        """
        _ob, info = self._env.reset()
        obs = {
            "reward": 0.0,
            "state": self._get_state(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self.render(),
            "info": info or {},
        }
        return obs

    def step(self, action):
        """
        与 DMC.step 类似：
        - action 是 dict，键为 "action"
        - 重复执行 _action_repeat 次底层 step
        - 汇总 reward，并根据 terminated / truncated 设置标志
        """
        act = action["action"]
        assert np.isfinite(act).all(), act

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        coordinates = None
        if hasattr(self._env, "get_xy"):
            coordinates = np.asarray(self._env.get_xy(), dtype=np.float32).copy()

        for _ in range(self._action_repeat):
            _ob, r, terminated, truncated, info = self._env.step(act)
            total_reward += float(r)
            if terminated or truncated:
                break

        if coordinates is not None and hasattr(self._env, "get_xy"):
            info = dict(info or {})
            info["coordinates"] = coordinates
            info["next_coordinates"] = np.asarray(self._env.get_xy(), dtype=np.float32).copy()

        obs = {
            "reward": total_reward,
            "state": self._get_state(),
            "is_first": False,
            "is_last": bool(terminated or truncated),
            "is_terminal": bool(terminated),
            "image": self.render(),
            "info": info or {},
        }
        return obs

    # ----------------- 透传其它接口（get_xy / set_goal 等） -----------------

    def __getattr__(self, name):
        # 当 LocoMazeEnv 自身没有属性时，转发到底层 env
        return getattr(self._env, name)


# --------- 两个小壳子类：只是固定 domain，接口完全一致 ---------

class MazeEnv(LocoMazeEnv):
    """固定为普通迷宫任务（domain='maze'）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(domain="maze", *args, **kwargs)


class BallEnv(LocoMazeEnv):
    """固定为推球任务（domain='ball'）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(domain="ball", *args, **kwargs)
