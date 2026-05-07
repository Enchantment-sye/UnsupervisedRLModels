import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging

from envs import make_env
from envs.wrappers import NormalizeAction, TimeLimit, FrameStack
from load_pretrain_metra import load_pretrained_metra
from sac_finetune import FinetuneSACAgent
from workers.rollout import SkillRolloutWorker

def visualize_tsne(args):
    # 1. Setup Environment
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('VisualizetSNE')
    
    # Env
    if args.task == 'd4rl_kitchen':
         try:
             import sys
             sys.path.append(os.getcwd())
             from mykitchen_opt import MyKitchenEnv
             print("Using optimized MyKitchenEnv")
             env = MyKitchenEnv(
                action_repeat=args.action_repeat,
                width = args.render_size,
                use_pixel=args.use_encoder
             )
             env = NormalizeAction(env)
             env = TimeLimit(env, args.time_limit)
             if args.framestack > 1:
                 env = FrameStack(env, k=args.framestack)
         except Exception as e:
             print(f"Failed to import optimized env: {e}, falling back to default")
             env = make_env(mode="eval", config=args)
    else:
        env = make_env(mode="eval", config=args)
    
    # 2. Load Agent/Policy
    device = torch.device(args.device)
    pre_metra = load_pretrained_metra(
        os.path.dirname(args.skill_policy_path),
        device=device,
        skill_policy_name=os.path.basename(args.skill_policy_path),
        load_traj_encoder=False,
        freeze=True, 
        eval_mode=True
    )
    
    skill_policy = pre_metra.skill_policy
    dim_skill = pre_metra.dim_skill
    discrete_skill = pre_metra.discrete
    
    # 3. Collect Data
    logger.info(f"Collecting {args.num_trajs} trajectories...")
    rollout_worker = SkillRolloutWorker(seed=args.seed, time_limit=args.time_limit, cur_extra_keys=['skill'])
    
    all_obs = []
    all_skills = []
    all_timesteps = []
    
    for i in tqdm(range(args.num_trajs)):
        # Sample a skill
        if discrete_skill:
            skill_idx = i % dim_skill
            skill = np.eye(dim_skill)[skill_idx]
            skill_label = skill_idx
        else:
            skill = np.random.randn(dim_skill)
            skill = skill / np.linalg.norm(skill)
            skill_label = i  # Continuous skills don't have discrete labels easily, use traj index
            
        batch = rollout_worker.rollout(
            env, 
            skill_policy, 
            extra={'skill': skill}, 
            deterministic_policy=True
        )
        
        # batch['obs'] is list of obs (or dicts if not flattened/processed yet)
        # If it's a dict env, rollout_worker might return dicts
        # Let's handle 'image' key if present
        
        trajs = batch.to_trajectory_list()
        for traj in trajs:
            obs_seq = traj['observations'] # Shape (T, ...)
            
            # If obs is image (T, C, H, W) or (T, H, W, C)
            # Check shape
            if obs_seq.ndim > 2:
                # Likely image
                pass
            
            all_obs.append(obs_seq)
            all_skills.extend([skill_label] * len(obs_seq))
            all_timesteps.extend(list(range(len(obs_seq))))

    # Concatenate all observations
    # Handle if obs are different lengths (they shouldn't be if successful, but could be)
    all_obs_cat = np.concatenate(all_obs, axis=0)
    
    logger.info(f"Collected {all_obs_cat.shape[0]} samples. Shape: {all_obs_cat.shape}")
    
    # 4. Extract Features
    if args.mode == 'feature':
        logger.info("Extracting features using encoder...")
        # Assuming agent has encoder. metra.skill_policy might have it.
        # pre_metra.skill_policy is PolicyEx(module=WithEncoder(encoder, mlp))
        if hasattr(skill_policy, '_module') and hasattr(skill_policy._module, 'encoder'):
            encoder = skill_policy._module.encoder
            
            # Process in batches to avoid OOM
            batch_size = 256
            features_list = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(all_obs_cat), batch_size)):
                    batch_obs = all_obs_cat[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch_obs).float().to(device)
                    
                    # Preprocess if needed (e.g. normalize 0-255 to 0-1 if encoder expects it?)
                    # Metra encoder usually expects what env returns.
                    # If env returns 0-255 uint8, and encoder expects float, we might need /255.
                    # Usually make_env handles this (e.g. ToTensor). 
                    # Assuming env returns proper format.
                    
                    # Flatten if encoder expects (B, C*H*W) or (B, C, H, W)
                    # Encoder in networks.py:
                    # def forward(self, obs): ... return self.trunk(obs)
                    # If obs is (B, C, H, W), Conv2d works.
                    # If obs is (B, flat), we need to reshape.
                    
                    if args.pixel_shape and batch_tensor.ndim == 2:
                        # Reshape flattened obs back to (B, C, H, W)
                        C, H, W = args.pixel_shape
                        batch_tensor = batch_tensor.view(-1, C, H, W)
                    
                    feat = encoder(batch_tensor)
                    features_list.append(feat.cpu().numpy())
            
            data_to_tsne = np.concatenate(features_list, axis=0)
        else:
            logger.warning("Encoder not found in policy! Falling back to raw flattened images.")
            data_to_tsne = all_obs_cat.reshape(all_obs_cat.shape[0], -1)
            
    else: # mode == 'raw'
        logger.info("Using raw flattened images...")
        data_to_tsne = all_obs_cat.reshape(all_obs_cat.shape[0], -1)
        
        # Subsample if raw images are too huge and many
        if data_to_tsne.shape[1] > 10000:
             logger.warning("Dimensionality very high, PCA might be slow. Consider --mode feature.")

    # 5. Run t-SNE
    logger.info("Running t-SNE...")
    # Subsample points for t-SNE speed if too many
    max_points = 5000
    if data_to_tsne.shape[0] > max_points:
        logger.info(f"Subsampling to {max_points} points for visualization speed...")
        indices = np.random.choice(data_to_tsne.shape[0], max_points, replace=False)
        data_to_tsne = data_to_tsne[indices]
        plot_skills = np.array(all_skills)[indices]
        plot_timesteps = np.array(all_timesteps)[indices]
    else:
        plot_skills = np.array(all_skills)
        plot_timesteps = np.array(all_timesteps)

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    tsne_results = tsne.fit_transform(data_to_tsne)
    
    # 6. Plot
    logger.info("Plotting...")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Colored by Skill
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=plot_skills, cmap='tab10' if discrete_skill else 'viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Skill ID')
    plt.title('t-SNE colored by Skill')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    # Plot 2: Colored by Time Step
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=plot_timesteps, cmap='plasma', alpha=0.6, s=10)
    plt.colorbar(scatter2, label='Time Step')
    plt.title('t-SNE colored by Time Step')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    save_path = os.path.join(args.work_dir, f'tsne_{args.mode}_{args.task}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")
    print(f"Visualization complete. Saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--work_dir', type=str, default='vis_results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--camera', type=str, default='corner')
    parser.add_argument('--dmc_camera', type=int, default=-1)
    parser.add_argument('--time_limit', type=int, default=50) # Shorter for vis
    parser.add_argument('--flatten_obs', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=3)
    
    # Metra args
    parser.add_argument('--skill_policy_path', type=str, required=True)
    parser.add_argument('--dim_skill', type=int, default=None)
    parser.add_argument('--discrete_skill', action='store_true')
    parser.add_argument('--use_encoder', action='store_true')
    parser.add_argument('--pixel_shape', type=str, default=None) 
    parser.add_argument('--device', type=str, default='cuda')
    
    # Vis args
    parser.add_argument('--num_trajs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'raw'], help='Visualize encoder features or raw pixels')
    
    args = parser.parse_args()
    
    if args.pixel_shape:
        args.pixel_shape = tuple(map(int, args.pixel_shape.split(',')))
    
    os.makedirs(args.work_dir, exist_ok=True)
    visualize_tsne(args)
