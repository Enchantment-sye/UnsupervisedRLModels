import os
import gymnasium
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import gym
import numpy as np
import cv2
import mujoco


# all tasks
from bigym.envs.reach_target import ReachTarget, ReachTargetSingle, ReachTargetDual
from bigym.envs.pick_and_place import PutCups, TakeCups, StoreBox, PickBox, SaucepanToHob, StoreKitchenware, ToastSandwich, FlipSandwich, RemoveSandwich
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
from bigym.envs.dishwasher import DishwasherOpen, DishwasherClose, DishwasherOpenTrays, DishwasherCloseTrays
from bigym.envs.dishwasher_cups import DishwasherLoadCups, DishwasherUnloadCups, DishwasherUnloadCupsLong
from bigym.envs.dishwasher_plates import DishwasherLoadPlates, DishwasherUnloadPlates, DishwasherUnloadPlatesLong
from bigym.envs.dishwasher_cutlery import DishwasherLoadCutlery, DishwasherUnloadCutlery, DishwasherUnloadCutleryLong
from bigym.envs.cupboards import DrawerTopOpen, DrawerTopClose, DrawersAllOpen, DrawersAllClose, WallCupboardOpen, WallCupboardClose, CupboardsOpenAll, CupboardsCloseAll
# obs, acs
from bigym.action_modes import TorqueActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig


def _patch_bigym_renderer():
    from bigym.bigym_renderer import BiGymRenderer, BiGymWindowViewer
    from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

    if getattr(BiGymRenderer, "_metra_patch_applied", False):
        return

    def _get_viewer(self, render_mode: str):
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = BiGymWindowViewer(self.model, self.data)
            elif render_mode in {"rgb_array", "depth_array"}:
                width = getattr(self, "width", None) or 640
                height = getattr(self, "height", None) or 480
                self.viewer = OffScreenViewer(self.model, self.data, width, height)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, or depth_array"
                )
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        self.viewer.make_context_current()
        return self.viewer

    BiGymRenderer._get_viewer = _get_viewer
    BiGymRenderer.get_viewer = _get_viewer
    BiGymRenderer._metra_patch_applied = True


_patch_bigym_renderer()

task_map_dict = {
    # Reach Target Tasks
    "reach_target": ReachTarget,
    "reach_target_single": ReachTargetSingle,
    "reach_target_dual": ReachTargetDual,

    # Pick and Place Tasks
    "put_cups": PutCups,
    "take_cups": TakeCups,
    "store_box": StoreBox,
    "pick_box": PickBox,
    "saucepan_to_hob": SaucepanToHob,
    "store_kitchenware": StoreKitchenware,
    "toast_sandwich": ToastSandwich,
    "flip_sandwich": FlipSandwich,
    "remove_sandwich": RemoveSandwich,

    # Move Plates Tasks
    "move_plate": MovePlate,
    "move_two_plates": MoveTwoPlates,

    # Manipulation Tasks
    "flip_cup": FlipCup,
    "flip_cutlery": FlipCutlery,
    "stack_blocks": StackBlocks,

    # Groceries Tasks
    "groceries_store_lower": GroceriesStoreLower,
    "groceries_store_upper": GroceriesStoreUpper,

    # Dishwasher Tasks (General)
    "dishwasher_open": DishwasherOpen,
    "dishwasher_close": DishwasherClose,
    "dishwasher_open_trays": DishwasherOpenTrays,
    "dishwasher_close_trays": DishwasherCloseTrays,

    # Dishwasher Cups Tasks
    "dishwasher_load_cups": DishwasherLoadCups,
    "dishwasher_unload_cups": DishwasherUnloadCups,
    "dishwasher_unload_cups_long": DishwasherUnloadCupsLong,

    # Dishwasher Plates Tasks
    "dishwasher_load_plates": DishwasherLoadPlates,
    "dishwasher_unload_plates": DishwasherUnloadPlates,
    "dishwasher_unload_plates_long": DishwasherUnloadPlatesLong,

    # Dishwasher Cutlery Tasks
    "dishwasher_load_cutlery": DishwasherLoadCutlery,
    "dishwasher_unload_cutlery": DishwasherUnloadCutlery,
    "dishwasher_unload_cutlery_long": DishwasherUnloadCutleryLong,

    # Cupboards Tasks
    "drawer_top_open": DrawerTopOpen,
    "drawer_top_close": DrawerTopClose,
    "drawers_all_open": DrawersAllOpen,
    "drawers_all_close": DrawersAllClose,
    "wall_cupboard_open": WallCupboardOpen,
    "wall_cupboard_close": WallCupboardClose,
    "cupboards_open_all": CupboardsOpenAll,
    "cupboards_close_all": CupboardsCloseAll,
}



class BiGymEnv:
    def __init__(
            self,
            name,
            seed=None,
            action_repeat=1,
            size=(64, 64),
            camera='head',
            flatten_obs=False,
            freeze_lower_body=True,
    ):

        self._env = task_map_dict[name](
            action_mode=TorqueActionMode(floating_base=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name=camera,
                        rgb=True,
                        depth=False,
                        resolution=size,
                    )
                ],
            ),
            render_mode='rgb_array',
        )
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self.flatten_obs = flatten_obs

        self.freeze_lower_body = bool(freeze_lower_body)
        self._base_action_dim = 0
        try:
            am = getattr(self._env, "action_mode", None)
            if am is not None and getattr(am, "floating_base", False):
                floating_dofs = getattr(am, "floating_dofs", [])
                self._base_action_dim = int(len(floating_dofs))
        except Exception:
            # 推断失败就不冻结（安全退化）
            self._base_action_dim = 0
        self._prev_state = None

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "info": dict(),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        u = np.asarray(action["action"], dtype=np.float32).copy()
        assert np.isfinite(u).all(), u
        prev_state = self._prev_state

        # --- 关键：冻结 floating base（下半身）---
        # bigym 4.1.0 默认 floating_dofs = [X, Y, RZ] => _base_action_dim=3
        if self.freeze_lower_body and self._base_action_dim > 0:
            u[: self._base_action_dim] = 0.0

        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state_dict, rew, done, _, info = self._env.step(u)
            success = float(info["task_success"])
            reward += rew or 0.0
            if done or success == 1.0:
                break

        assert success in [0.0, 1.0]
        image = self.render()
        next_state = np.asarray(self.bigym_lowdim_state(state_dict))
        self._prev_state = next_state
        obs = {
            "state":  self._prev_state,
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": image,
            "success": success,
            "info": {
                "success": success,
                "state": self._prev_state
            },
        }
        return obs

    def reset(self):
        state_dict, info = self._env.reset()
        image = self.render()
        self._prev_state = self.bigym_lowdim_state(state_dict)
        obs = {
            "state":  self._prev_state,
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": image,
            "success": float(info["task_success"]),
            "info": {
                "state" : self._prev_state,
            },
        }
        return obs

    def render(self, mode='offscreen'):
        renderer = getattr(self._env, "mujoco_renderer", None)
        if renderer is not None:
            # Gymnasium's MujocoRenderer now requires an explicit target size for rgb_array rendering.
            renderer.width = self._size[1]
            renderer.height = self._size[0]
        obs = cv2.resize(self._env.render(), self._size)
        if self.flatten_obs:
            obs = obs.flatten()
        return obs

    def bigym_lowdim_state(self, obs: dict) -> np.ndarray:
        parts = []
        for k in ["proprioception", "proprioception_grippers", "proprioception_floating_base"]:
            if k in obs and obs[k] is not None:
                parts.append(np.asarray(obs[k]).ravel())
        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)

    


if __name__ == "__main__":

    def main():
        """
        Main function to test the BiGymEnv class with various scenarios.
        Tests include environment initialization, step functionality, reset, and rendering.
        """
        
        def test_env_initialization():
            """Test environment initialization with different parameters."""
            print("=== Testing Environment Initialization ===")
            
            try:
                # Test with default parameters
                env = BiGymEnv("reach_target")
                print("✓ Default initialization successful")
                
                # Test with custom parameters
                env_custom = BiGymEnv(
                    name="pick_box", 
                    seed=42, 
                    action_repeat=2, 
                    size=(128, 128),
                    camera='head',
                    flatten_obs=True
                )
                print("✓ Custom initialization successful")
                
                # Test observation and action spaces
                obs_space = env.obs_space
                act_space = env.act_space
                print(f"✓ Observation space keys: {list(obs_space.keys())}")
                print(f"✓ Action space shape: {act_space['action'].shape}")
                
            except Exception as e:
                print(f"✗ Initialization failed: {e}")
        
        def test_environment_workflow():
            """Test complete environment workflow: reset -> step -> render."""
            print("\n=== Testing Environment Workflow ===")
            
            try:
                env = BiGymEnv("reach_target", size=(64, 64))
                
                # Test reset
                obs = env.reset()
                print("✓ Environment reset successful")
                print(f"  - Initial observation keys: {list(obs.keys())}")
                print(f"  - Image shape: {obs['image'].shape}")
                print(f"  - Is first: {obs['is_first']}")
                print(f"  - Initial success: {obs['success']}")
                
                # Test step with random action
                action_dim = env.act_space["action"].shape[0]
                print(f"  - Action Space: {env.act_space['action'].low}, {env.act_space['action'].high}")
                random_action = env.act_space["action"].sample()
                
                step_obs = env.step({"action": random_action})
                print("✓ Environment step successful")
                print(f"  - Reward: {step_obs['reward']}")
                print(f"  - Is first: {step_obs['is_first']}")
                print(f"  - Success: {step_obs['success']}")
                
            except Exception as e:
                print(f"✗ Workflow test failed: {e}")
        
        def test_multiple_steps():
            """Test multiple environment steps and check for consistency."""
            print("\n=== Testing Multiple Steps ===")
            
            try:
                env = BiGymEnv("flip_cup", action_repeat=1)
                obs = env.reset()
                
                total_reward = 0
                num_steps = 5
                action_dim = env.act_space["action"].shape[0]
                
                print(f"Running {num_steps} steps...")
                for i in range(num_steps):
                    # Generate random action within valid range
                    action = env.act_space["action"].sample()
                    obs = env.step({"action": action})
                    total_reward += obs['reward']
                    
                    print(f"  Step {i+1}: reward={obs['reward']:.4f}, success={obs['success']}")
                    
                    # Break if task is successful
                    if obs['success'] == 1.0:
                        print(f"  ✓ Task completed successfully at step {i+1}!")
                        break
                
                print(f"✓ Total reward accumulated: {total_reward:.4f}")
                
            except Exception as e:
                print(f"✗ Multiple steps test failed: {e}")
        
        def test_different_tasks():
            """Test initialization and basic functionality of different tasks."""
            print("\n=== Testing Different Tasks ===")
            
            test_tasks = [
                "reach_target",
                "put_cups", 
                "move_plate",
                "flip_cup",
                "dishwasher_open"
            ]
            
            for task_name in test_tasks:
                try:
                    env = BiGymEnv(task_name, size=(32, 32))  # Smaller size for faster testing
                    obs = env.reset()
                    
                    # Test one step
                    action_dim = env.act_space["action"].shape[0]
                    action = np.zeros(action_dim)  # Zero action for safety
                    step_obs = env.step({"action": action})
                    
                    print(f"✓ Task '{task_name}' working properly")
                    
                except Exception as e:
                    print(f"✗ Task '{task_name}' failed: {e}")
        
        def test_rendering_options():
            """Test different rendering configurations."""
            print("\n=== Testing Rendering Options ===")
            
            try:
                # Test normal rendering
                env1 = BiGymEnv("reach_target", size=(64, 64), flatten_obs=False)
                obs1 = env1.reset()
                print(f"✓ Normal rendering - Image shape: {obs1['image'].shape}")
                
                # Test flattened observation
                env2 = BiGymEnv("reach_target", size=(32, 32), flatten_obs=True)
                obs2 = env2.reset()
                print(f"✓ Flattened rendering - Image shape: {obs2['image'].shape}")
                
                # Test different image sizes
                sizes = [(48, 48), (96, 96)]
                for size in sizes:
                    env = BiGymEnv("reach_target", size=size)
                    obs = env.reset()
                    expected_shape = size + (3,)
                    actual_shape = obs['image'].shape
                    assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} vs {expected_shape}"
                    print(f"✓ Size {size} - Image shape: {actual_shape}")
                    
            except Exception as e:
                print(f"✗ Rendering test failed: {e}")
        
        def test_action_validation():
            """Test action validation and edge cases."""
            print("\n=== Testing Action Validation ===")
            
            try:
                env = BiGymEnv("reach_target")
                env.reset()
                action_dim = env.act_space["action"].shape[0]
                
                # Test valid action
                low, high = env.act_space["action"].low, env.act_space["action"].high
                print(f"  - Action Space: {low}, {high}")
                valid_action = np.random.uniform(low, high, action_dim)
                obs = env.step({"action": valid_action})
                print("✓ Valid action processed successfully")
                
                # Test edge case: zero action
                zero_action = np.zeros(action_dim)
                obs = env.step({"action": zero_action})
                print("✓ Zero action processed successfully")
                
                # Test invalid action (should raise assertion error)
                try:
                    invalid_action = np.full(action_dim, np.inf)
                    env.step({"action": invalid_action})
                    print("✗ Invalid action should have failed")
                except AssertionError:
                    print("✓ Invalid action properly rejected")
                    
            except Exception as e:
                print(f"✗ Action validation test failed: {e}")

        # Run all tests
        print("Starting BiGymEnv Testing Suite...\n")
        
        test_env_initialization()
        test_environment_workflow()
        test_multiple_steps()
        test_different_tasks()
        test_rendering_options()
        test_action_validation()
        
        print("\n=== Testing Complete ===")
        print("Check the output above for any failed tests (marked with ✗)")

        return
    
    main()