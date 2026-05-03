import numpy as np
import gymnasium as gym
import glob
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.pendulum_custom import TargetAnglePendulum

def evaluate_untrained_agent(env, num_episodes=20):
    """Simulates the Step 0 baseline using a random policy"""
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0
        while not done:
            # An untrained network outputs random noise
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return np.mean(returns)

def patch_all_files():
    # Find all .npy files in the current folder
    files = glob.glob("*.npy")
    
    for filename in files:
        data = np.load(filename)
        
        # Only patch files that have exactly 10 columns (missing Step 0)
        if len(data.shape) == 2 and data.shape[1] == 10:
            print(f"Patching: {filename}")
            
            # Extract the target angle from the filename
            match = re.search(r'angle_(-?\d+)', filename)
            if not match:
                continue
            angle = int(match.group(1))
            
            # Setup the environment for this specific angle
            base_env = gym.make('Pendulum-v1', max_episode_steps=1000)
            env = TargetAnglePendulum(base_env, target_angle=angle, reward_scale=1.0)
            
            step_zero_data = []
            for seed in range(15):
                env.action_space.seed(seed)
                ret = evaluate_untrained_agent(env)
                step_zero_data.append(ret)
                
            # Convert to a column vector (15, 1)
            step_zero_col = np.array(step_zero_data).reshape(15, 1)
            
            # Glue the new Step 0 column to the front of the 10 existing columns
            new_data = np.hstack((step_zero_col, data))
            
            # Overwrite the file with the new 11-column data
            np.save(filename, new_data)
            print(f"--> Successfully updated to shape {new_data.shape}\n")
        else:
            print(f"Skipping {filename} (Already patched or wrong shape: {data.shape})")

if __name__ == "__main__":
    patch_all_files()