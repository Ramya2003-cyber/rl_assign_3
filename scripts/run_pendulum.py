import numpy as np
import torch
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.pendulum_custom import TargetAnglePendulum
from agent.sac import SACAgent
import gymnasium as gym
from replay_buffer import ReplayBuffer
import hydra

# Tiny mock logger to prevent SAC from crashing when it tries to log metrics
class DummyLogger:
    def log(self, *args, **kwargs): pass
    def log_histogram(self, *args, **kwargs): pass
    def log_param(self, *args, **kwargs): pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(agent, env, num_episodes=20):
    returns = []
    for i in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(obs, sample=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            obs = next_obs
        returns.append(episode_reward)
    return np.mean(returns)

@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg):
    all_results = []
    dummy_logger = DummyLogger() # Instantiate the safety logger

    for seed in range(1, 16):
        set_seed(seed)
        
        # CREATE TRAINING ENV (Fixed typo to max_episode_steps)
        base_env = gym.make('Pendulum-v1', max_episode_steps=1000)
        env = TargetAnglePendulum(base_env, target_angle=cfg.custom.target_angle, reward_scale=cfg.custom.reward_scale)
        
        # CREATE DEDICATED EVALUATION ENV
        base_eval_env = gym.make('Pendulum-v1', max_episode_steps=1000)
        eval_env = TargetAnglePendulum(base_eval_env, target_angle=cfg.custom.target_angle, reward_scale=cfg.custom.reward_scale)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]
        
        replay_buf = ReplayBuffer(obs_dim, action_dim, cfg.replay_buffer_capacity, cfg.device)
        
        sac_agent = hydra.utils.instantiate(
            cfg.agent, 
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            action_range=action_range,
            learnable_temperature=cfg.custom.auto_tune,
            init_temperature=cfg.custom.alpha
        )
        
        obs, info = env.reset(seed=seed)
        seed_evals = []
        
        for i in range(1, 100001):
            action = sac_agent.act(obs, sample=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buf.add(obs, action, reward, next_obs, done)
            obs = next_obs
            
            if i >= cfg.agent.batch_size:
                # Pass the dummy logger instead of None
                sac_agent.update(replay_buf, dummy_logger, i)
                
            if done:
                obs, info = env.reset()
                
            if i % 10000 == 0:
                # Use the dedicated eval_env here!
                avg_return = evaluate(sac_agent, eval_env, num_episodes=20)
                seed_evals.append(avg_return)
                print(f"Seed {seed}, Step {i}, Avg Return: {avg_return}")
                
        all_results.append(seed_evals)

    # Dynamic save
    filename = f"results_angle_{cfg.custom.target_angle}_auto_{cfg.custom.auto_tune}_alpha_{cfg.custom.alpha}_scale_{cfg.custom.reward_scale}.npy"
    np.save(filename, all_results)
    print(f"Saved {filename}")

if __name__ == '__main__':
    main()