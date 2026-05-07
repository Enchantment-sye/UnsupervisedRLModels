import os
import cv2
import gym
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

from envs.color_gradients import sample_gradient_rgb_uint8
from envs.locomotion_coverage import compute_locomotion_coverage_metrics


def _read_video_frames(fname, grayscale=False):
    try:
        import skvideo.io
    except ImportError as exc:
        raise ImportError(
            "skvideo is required for RandomVideoSource video backgrounds. "
            "Install sk-video or disable video-background features."
        ) from exc
    if grayscale:
        return skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
    return skvideo.io.vread(fname)


class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, flatten_obs=False, render_image=True):
        os.environ.setdefault("MUJOCO_GL", "egl")
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._domain = domain
        self._task = task

        self._env = None

        # 对 cheetah / quadruped / humanoid 尝试使用 METRA 的 custom_dmc_tasks
        if domain in ["quadruped", "cheetah", "humanoid"]:
            # try:
            if domain == "quadruped":
                from envs.custom_dmc_tasks import quadruped as dm_module
            elif domain == "cheetah":
                from envs.custom_dmc_tasks import cheetah as dm_module
            else:
                from envs.custom_dmc_tasks import humanoid as dm_module

            env_kwargs = dict(flat_observation=True)

            if hasattr(dm_module, "SUITE") and task in dm_module.SUITE:
                self._env = dm_module.SUITE[task](
                    environment_kwargs=env_kwargs
                )
            elif hasattr(dm_module, task):
                fn = getattr(dm_module, task)
                self._env = fn(environment_kwargs=env_kwargs)
            else:
                # 找不到自定义任务就回退
                self._env = None
            # except Exception as e:
            #     print(f"[DMC] custom_dmc_tasks {domain}_{task} failed: {e!r}")
            #     self._env = None

        if self._env is None:
            if domain == "manip":
                from dm_control import manipulation

                self._env = manipulation.load(task + "_vision")
            elif domain == "locom":
                from dm_control.locomotion.examples import basic_rodent_2020

                self._env = getattr(basic_rodent_2020, task)()
            else:
                from dm_control import suite

                self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        self._render_image = bool(render_image)
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                quadruped_run_forward_color=2,
                pentaped_walk=2,
                pentaped_run=2,
                pentaped_escape=2,
                pentaped_fetch=2,
                biped_walk=2,
                biped_run=2,
                biped_escape=2,
                biped_fetch=2,
                triped_walk=2,
                triped_run=2,
                triped_escape=2,
                triped_fetch=2,
                hexaped_walk=2,
                hexaped_run=2,
                hexaped_escape=2,
                hexaped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)
        self.flatten_obs = flatten_obs
        self._apply_metra_colors()

    def _get_torso_xyz(self):
        if not hasattr(self._env, "physics"):
            return None
        if self._domain not in ("cheetah", "quadruped", "humanoid"):
            return None
        try:
            xyz = self._env.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        except Exception:
            return None
        return np.asarray(xyz, dtype=np.float32).reshape(-1)

    def _build_coordinate_info(self, xyz_before, xyz_after):
        if xyz_before is None or xyz_after is None:
            return {}
        if self._domain == "cheetah":
            return {
                "coordinates": np.array([xyz_before[0], 0.0], dtype=np.float32),
                "next_coordinates": np.array([xyz_after[0], 0.0], dtype=np.float32),
            }
        if self._domain in ("quadruped", "humanoid"):
            return {
                "coordinates": np.array([xyz_before[0], xyz_before[1]], dtype=np.float32),
                "next_coordinates": np.array([xyz_after[0], xyz_after[1]], dtype=np.float32),
            }
        return {}

    def _apply_metra_colors(self):
        """
        完全仿照 envs.custom_dmc_tasks.pixel_wrappers.RenderWrapper：

        - cheetah：沿一个轴用 rainbow colormap 上色
        - quadruped / humanoid：2D 线性渐变
        """
        if not hasattr(self._env, "physics"):
            return

        import numpy as np
        import matplotlib.pyplot as plt

        physics = self._env.physics
        model = physics.model
        required_texture_attrs = ("tex_type", "tex_height", "tex_width", "tex_adr", "tex_rgb")
        if not all(hasattr(model, attr) for attr in required_texture_attrs):
            return
        n_tex = len(model.tex_type)

        if self._domain == "cheetah":
            # cheetah：rainbow 颜色条
            for i in range(n_tex):
                if model.tex_type[i] == 0:  # 2D texture
                    height = model.tex_height[i]
                    width = model.tex_width[i]
                    s = model.tex_adr[i]
                    colors = []
                    for y in range(width):
                        scaled_y = np.clip((y / width - 0.5) * 4 + 0.5, 0, 1)
                        rgb = (np.array(plt.cm.rainbow(scaled_y))[:3] * 255).astype(
                            np.uint8
                        )
                        colors.append(rgb)
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            model.tex_rgb[cur_s : cur_s + 3] = colors[y]
        else:
            # 其它（quadruped / humanoid）：二维渐变
            for i in range(n_tex):
                if model.tex_type[i] == 0:
                    height = model.tex_height[i]
                    width = model.tex_width[i]
                    s = model.tex_adr[i]
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            model.tex_rgb[cur_s : cur_s + 3] = sample_gradient_rgb_uint8(
                                "dmc_quadruped_run_forward_color",
                                x / max(height, 1),
                                y / max(width, 1),
                            )

        # 和 RenderWrapper 一样：材质重复次数设为 1，避免花纹太密
        if hasattr(model, "mat_texrepeat"):
            model.mat_texrepeat[:, :] = 1

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, self._env.physics.get_state().shape, dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "info": {
                "state": gym.spaces.Box(-np.inf, np.inf, self._env.physics.get_state().shape, dtype=np.float32)
            },
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        xyz_before = self._get_torso_xyz()
        for _ in range(self._action_repeat):
            time_step = self._env.step(action["action"])
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        xyz_after = self._get_torso_xyz()
        assert time_step.discount in (0, 1)
        info = {
            "state": self._env.physics.get_state().copy(),
        }
        info.update(self._build_coordinate_info(xyz_before, xyz_after))
        obs = {
            "reward": reward,
            "state": self._env.physics.get_state().copy(),
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self.render(),
            "info": info,
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "state": self._env.physics.get_state().copy(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self.render(),
            "info": {
                "state": self._env.physics.get_state().copy(),
            },
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def calc_eval_metrics(self, trajectories, is_option_trajectories=False):
        coord_dim = 2 if self._domain in ("quadruped", "humanoid") else 1
        return compute_locomotion_coverage_metrics(trajectories, coord_dim)
    
    def render(self, mode='offscreen'):
        if not self._render_image:
            image = np.zeros(self._size + (3,), dtype=np.uint8)
            return image.flatten() if self.flatten_obs else image
        obs = self._env.physics.render(*self._size, camera_id=self._camera)
        if self.flatten_obs:
            obs = obs.flatten()
        return obs


class RandomVideoSource:
    def __init__(self, env, video_dir, shape, total_frames=None, grayscale=False):
        """
        Args:
            filelist: a list of video files
        """
        self._env = env
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]
        self.build_arr()
        self.current_idx = 0
        self._pixels_key = 'image'
        self.reset()

    def build_arr(self):
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                frames = _read_video_frames(fname, grayscale=self.grayscale)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0: random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    frames = _read_video_frames(fname, grayscale=self.grayscale)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames: break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None] ## THIS IS NOT A BUG! cv2 uses (width, height)
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0])) 
                        pbar.update(1)
                        total_frame_i += 1


    def step(self, action):
        obs = self._env.step(action)
        pixels = self._extract_pixels(obs)
        img = self.get_obs(pixels)
        obs['image'] = img
        return obs

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)
        obs = self._env.reset()
        pixels = self._extract_pixels(obs)
        img = self.get_obs(pixels)
        obs['image'] = img
        return obs

    def get_obs(self, img):
        img = img.transpose(1, 2, 0)
        mask = np.logical_and((img[:, :, 2] > img[:, :, 1]), (img[:, :, 2] > img[:, :, 0]))  # hardcoded for dmc
        bg = self.get_image()
        img[mask] = bg[mask]
        # img = img.transpose(2, 0, 1).copy()
        img = img.copy()
        # CHW to HWC for tensorflow
        return img

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def act_space(self):
        return self._env.act_space

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def _extract_pixels(self, obs):
        pixels = obs[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()
