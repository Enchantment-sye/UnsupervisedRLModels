import os
import imageio.v2 as imageio
import numpy as np

from envs.maze.maze import make_maze_env


def rollout_random_ant_maze_video(
        output_path="ant_maze_random.mp4",
        maze_type="large",
        episode_steps=400,
        fps=10,
        width=256,
        height=256,
):
    """
    在 Ant + Maze 环境中用随机策略跑一条轨迹，并把画面保存为 mp4 视频。

    Args:
        output_path: 输出视频文件路径（.mp4）
        maze_type: 'arena' / 'medium' / 'large' / 'giant' / 'teleport'
        episode_steps: 每条轨迹的最大步数
        fps: 视频帧率
        width, height: 渲染分辨率
    """
    # 1) 创建 Ant + Maze 环境
    env = make_maze_env(
        loco_env_type="ant",
        maze_env_type="maze",
        maze_type=maze_type,
        ob_type="pixels",        # 观测用像素，更适合做视频
        render_mode="rgb_array", # AntEnv/MujocoEnv 的 render_mode
        width=width,
        height=height,
    )

    # 2) reset，拿到初始观测和 goal 图像
    obs, info = env.reset(options={"render_goal": True})
    print("obs shape:", obs.shape)
    if "goal" in info:
        print("goal obs shape:", info["goal"].shape)

    # 3) 准备视频写入器
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)

    # 4) 把初始帧写进去（可选）
    first_frame = env.render()   # MazeEnv.render() 返回 (H, W, 3) uint8
    writer.append_data(first_frame)

    # 5) 用随机动作 roll out 一条轨迹
    for t in range(episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 渲染当前画面
        frame = env.render()     # (H, W, 3) np.uint8
        writer.append_data(frame)

        if terminated or truncated:
            print(
                f"Episode finished at step {t}, "
                f"success={info.get('success', 0.0)}, reward={reward}"
            )
            break

    writer.close()
    env.close()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    rollout_random_ant_maze_video(
        output_path="videos/ant_maze_large_random.mp4",
        maze_type="large",
        episode_steps=600,
        fps=10,
        width=256,
        height=256,
    )
