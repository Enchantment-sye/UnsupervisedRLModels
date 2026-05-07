from envs.maze.test_maze_video import rollout_random_ant_maze_video


if __name__ == "__main__":
    rollout_random_ant_maze_video(
        output_path="videos/ant_maze_hetero_random_states.mp4",
        maze_type="hetero",
        episode_steps=600,
        fps=100,
        width=1024,
        height=1024,
    )
