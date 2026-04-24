import gymnasium as gym
import torch
import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.pendulum_custom import TargetAnglePendulum

@hydra.main(config_path="../config", config_name="train", version_base=None)
def visualize(cfg):
    # Use 'human' render mode to open the window
    base_env = gym.make('Pendulum-v1', render_mode='human')
    env = TargetAnglePendulum(base_env, target_angle=cfg.custom.target_angle)

    # Initialize agent (same as your training script)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cfg.agent.obs_dim = obs_dim
    cfg.agent.action_dim = action_dim
    cfg.agent.action_range = [-2.0, 2.0]

    agent = hydra.utils.instantiate(cfg.agent)

    # If you have saved weights, load them here:
    # agent.actor.load_state_dict(torch.load("actor_90.pth"))

    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            env.render() # This opens the pygame window
            action = agent.act(obs, sample=False) # sample=False for "Optimal" behavior
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode+1} finished. Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    visualize()