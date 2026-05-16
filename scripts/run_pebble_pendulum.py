import numpy as np
import torch
import torch.nn.functional as F
import random
import sys
import os
from omegaconf import OmegaConf
import gymnasium as gym
import hydra

# Allow importing from current directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import standard SAC and Buffers
from agent.sac import SACAgent
from agent.replay_buffer import ReplayBuffer

# Import Custom Environment
from envs.pendulum_custom import TargetAnglePendulum

# Import PEBBLE components
from agent.reward_net import RewardNet
from agent.preference_buffer import PreferenceBuffer
from envs.simulated_teacher import SimulatedTeacher
from core import utils

# Tiny mock logger to prevent SAC from crashing when it tries to log metrics
class DummyLogger:
    def log(self, *args, **kwargs): pass
    def log_histogram(self, *args, **kwargs): pass
    def log_param(self, *args, **kwargs): pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def get_ground_truth_reward(state, action, target_angle, reward_scale=1.0):
    """ Ground truth reward math matching TargetAnglePendulum exactly """
    cos_theta, sin_theta, angular_velocity = state
    rad_theta = np.arctan2(sin_theta, cos_theta)
    target_angle_rad = np.radians(target_angle)
    angular_error = (rad_theta - target_angle_rad + np.pi) % (2 * np.pi) - np.pi
    
    action_val = action[0] if isinstance(action, (np.ndarray, list)) else action
    custom_reward = -(angular_error**2 + 0.1 * angular_velocity**2 + 0.001 * action_val**2)
    return custom_reward * reward_scale

def pebble_update(agent, reward_net, replay_buf, logger, step):
    """ Performs standard SAC update but swaps environment reward for predicted reward. """
    obs, action, _, next_obs, not_done, not_done_no_max = replay_buf.sample(agent.batch_size)
    
    # Predict rewards with RewardNet
    with torch.no_grad():
        predicted_reward = reward_net(obs, action)
        
    # Standard SAC Critic Update
    agent.update_critic(obs, action, predicted_reward, next_obs, not_done_no_max, logger, step)

    # Standard SAC Actor & Alpha Update
    if step % agent.actor_update_frequency == 0:
        agent.update_actor_and_alpha(obs, logger, step)

    if step % agent.critic_target_update_frequency == 0:
        utils.soft_update_params(agent.critic, agent.critic_target, agent.critic_tau)


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg):
    dummy_logger = DummyLogger()
    
    seed = cfg.seed
    set_seed(seed)
    
    # -------------------------------------
    # COMPONENT INITIALIZATION
    # -------------------------------------
    
    # 1. Environment Initialization
    base_env = gym.make('Pendulum-v1', max_episode_steps=1000)
    env = TargetAnglePendulum(base_env, target_angle=cfg.custom.target_angle, reward_scale=cfg.custom.reward_scale)
    
    base_eval_env = gym.make('Pendulum-v1', max_episode_steps=1000)
    eval_env = TargetAnglePendulum(base_eval_env, target_angle=cfg.custom.target_angle, reward_scale=cfg.custom.reward_scale)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]
    
    device = torch.device(cfg.device)

    # 2. SAC Agent Initialization
    OmegaConf.set_struct(cfg, False) 
    cfg.agent.obs_dim = obs_dim
    cfg.agent.action_dim = action_dim
    cfg.agent.action_range = action_range
    
    sac_agent = hydra.utils.instantiate(
        cfg.agent, 
        learnable_temperature=cfg.custom.auto_tune,
        init_temperature=cfg.custom.alpha
    )

    replay_buf = ReplayBuffer(env.observation_space.shape, env.action_space.shape, int(cfg.replay_buffer_capacity), cfg.device)

    # 3. PEBBLE Components Initialization
    reward_net = RewardNet(state_dim=obs_dim, action_dim=action_dim).to(device)
    reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=3e-4)
    preference_buffer = PreferenceBuffer(capacity=10000)
    
    feedback_budget = getattr(cfg.custom, 'feedback_budget', 1000)
    teacher = SimulatedTeacher(feedback_budget=feedback_budget)
    
    # Segment Collector Settings
    segment_length = getattr(cfg.custom, 'segment_length', 50)
    all_segments = []
    recent_states = []
    recent_actions = []

    # Setup saving and evaluations
    eval_results = []
    filename = f"pebble_results_angle_{cfg.custom.target_angle}_budget_{feedback_budget}_seed_{seed}.npy"
    obs, info = env.reset(seed=seed)
    num_seed_steps = cfg.num_seed_steps # (Phase 1 Steps)

    # INITIAL EVALUATION
    avg_return = evaluate(sac_agent, eval_env, num_episodes=20)
    eval_results.append(avg_return)
    print(f"Step 0, Eval Return (True Reward): {avg_return:.2f}, Budget Left: {teacher.feedback_budget}")

    # -------------------------------------
    # MAIN TRAINING LOOP (INTERLEAVED)
    # -------------------------------------
    for step in range(1, cfg.num_train_steps + 1):
        
        # 1. Action Selection
        if step <= num_seed_steps:
            action = env.action_space.sample()  # Phase 1: Random acts
        else:
            action = sac_agent.act(obs, sample=True) # Phase 3: Agent acts
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 2. Add transition to SAC buffer
        replay_buf.add(obs, action, reward, next_obs, done, terminated)
        
        # 3. The Segment Collector
        recent_states.append(obs)
        recent_actions.append(action)
        if len(recent_states) == segment_length:
            all_segments.append((np.array(recent_states), np.array(recent_actions)))
            recent_states = []
            recent_actions = []

        obs = next_obs
        if done:
            obs, info = env.reset()

        # -------------------------------------
        # PHASE 2: PRE-TRAIN REWARD NET
        # -------------------------------------
        if step == num_seed_steps:
            initial_queries = int(feedback_budget * 0.1)
            print(f"--- Phase 2: Pre-training RewardNet with {initial_queries} queries ---")
            
            for _ in range(initial_queries):
                if len(all_segments) < 2: break
                idx1, idx2 = np.random.choice(len(all_segments), 2, replace=False)
                seg_A = all_segments[idx1]
                seg_B = all_segments[idx2]
                
                def env_reward_fn(s, a):
                    return get_ground_truth_reward(s, a, cfg.custom.target_angle, cfg.custom.reward_scale)
                    
                label = teacher.evaluate_preference(seg_A, seg_B, env_reward_fn)
                if label is not None:
                    preference_buffer.store(seg_A, seg_B, label)
            
            # Train RewardNet on the initial buffer
            reward_net.train()
            batch_size = min(256, len(preference_buffer))
            if len(preference_buffer) > 0:
                for _ in range(50): # 50 epochs of pre-training
                    seg_A, seg_B, labels = preference_buffer.sample(batch_size, device=device)
                    loss = reward_net.preference_loss(seg_A, seg_B, labels)
                    reward_optimizer.zero_grad()
                    loss.backward()
                    reward_optimizer.step()
                
            print("--- RewardNet Pre-training Complete ---")

        # -------------------------------------
        # PHASE 3: MAIN LOOP UPDATES
        # -------------------------------------
        if step > num_seed_steps:
            # 1. Update SAC using predicted rewards
            if step >= cfg.agent.batch_size:
                pebble_update(sac_agent, reward_net, replay_buf, dummy_logger, step)
                
            # 2. Interleaved Preference Queries every 1000 steps
            if step % 1000 == 0 and teacher.feedback_budget > 0:
                # Ask teacher for 10 new preference pairs
                queries_to_make = min(10, teacher.feedback_budget)
                for _ in range(queries_to_make): 
                    if len(all_segments) < 2: break
                    
                    idx1, idx2 = np.random.choice(len(all_segments), 2, replace=False)
                    seg_A = all_segments[idx1]
                    seg_B = all_segments[idx2]
                    
                    def env_reward_fn(s, a):
                        return get_ground_truth_reward(s, a, cfg.custom.target_angle, cfg.custom.reward_scale)
                        
                    label = teacher.evaluate_preference(seg_A, seg_B, env_reward_fn)
                    if label is not None:
                        preference_buffer.store(seg_A, seg_B, label)
                
                # Update RewardNet with new data
                reward_net.train()
                batch_size = min(256, len(preference_buffer))
                if len(preference_buffer) > 0:
                    for _ in range(10): # 10 epochs
                        seg_A, seg_B, labels = preference_buffer.sample(batch_size, device=device)
                        loss = reward_net.preference_loss(seg_A, seg_B, labels)
                        reward_optimizer.zero_grad()
                        loss.backward()
                        reward_optimizer.step()

        # -------------------------------------
        # THE EVALUATION TRAP
        # -------------------------------------
        if step % 10000 == 0:
            avg_return = evaluate(sac_agent, eval_env, num_episodes=20)
            eval_results.append(avg_return)
            print(f"Step {step}, Eval Return (True Reward): {avg_return:.2f}, Budget Left: {teacher.feedback_budget}")
            np.save(filename, np.array(eval_results))

    print(f"Training complete. Results saved to {filename}")

if __name__ == '__main__':
    main()
