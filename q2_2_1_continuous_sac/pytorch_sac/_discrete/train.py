#!/usr/bin/env python3
import numpy as np
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import gymnasium as gym

sys.path.append(os.getcwd())

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import hydra
from agent.sac import SACAgent
from agent.discrete_sac import DiscreteSACAgent 

def make_env(cfg):
    env = gym.make(cfg.env, render_mode='rgb_array' if cfg.save_video else None)
    env.reset(seed=cfg.seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        
        log_dir = os.path.join(self.work_dir, 'exp_local', cfg.env, f"{cfg.experiment}_seed_{cfg.seed}")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = Logger(log_dir, save_tb=cfg.log_save_tb, log_frequency=cfg.log_frequency, agent='sac')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        obs_dim = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            action_range = [0, action_dim - 1]
            print(f"--- INITIALIZING DISCRETE SAC AGENT ---")
            self.agent = DiscreteSACAgent(obs_dim, action_dim, action_range, self.device, cfg.agent.params.critic, 
                                          cfg.agent.params.actor, cfg.agent.params.discount, cfg.agent.params.init_temperature,
                                          cfg.agent.params.alpha_lr, cfg.agent.params.alpha_betas, cfg.agent.params.actor_lr,
                                          cfg.agent.params.actor_betas, cfg.agent.params.actor_update_frequency, 
                                          cfg.agent.params.critic_lr, cfg.agent.params.critic_betas, cfg.agent.params.critic_tau,
                                          cfg.agent.params.critic_target_update_frequency, cfg.agent.params.batch_size,
                                          cfg.agent.params.learnable_temperature)
            action_shape = (1,)
        else:
            action_dim = self.env.action_space.shape[0]
            action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]
            print(f"--- INITIALIZING CONTINUOUS SAC AGENT ---")
            self.agent = SACAgent(obs_dim, action_dim, action_range, self.device, cfg.agent.params.critic, 
                                  cfg.agent.params.actor, cfg.agent.params.discount, cfg.agent.params.init_temperature,
                                  cfg.agent.params.alpha_lr, cfg.agent.params.alpha_betas, cfg.agent.params.actor_lr,
                                  cfg.agent.params.actor_betas, cfg.agent.params.actor_update_frequency, 
                                  cfg.agent.params.critic_lr, cfg.agent.params.critic_betas, cfg.agent.params.critic_tau,
                                  cfg.agent.params.critic_target_update_frequency, cfg.agent.params.batch_size,
                                  cfg.agent.params.learnable_temperature)
            action_shape = self.env.action_space.shape

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, action_shape, int(self.cfg.replay_buffer_capacity), self.device)
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        
        self.step = 0
        self._next_eval_step = 0

    def evaluate(self):
        print(f"\n>>> EVALUATING AT STEP {self.step}...")
        avg_reward = 0
        for _ in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset(); done = False; episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent): action = self.agent.act(obs, sample=False)
                obs, reward, term, trunc, _ = self.env.step(action); done = term or trunc; episode_reward += reward
            avg_reward += episode_reward
        
        final_score = avg_reward / self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', final_score, self.step)
        self.logger.dump(self.step, ty='eval')
        print(f">>> EVAL FINISHED. REWARD: {final_score:.2f}\n")

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        
        while self.step < self.cfg.num_train_steps:
            if self.step % 100 == 0:
                print(f"Step: {self.step:6d} | Episode: {episode:4d} | R: {episode_reward:7.2f}", end="\r")

            if self.step >= self._next_eval_step:
                self.evaluate()
                self._next_eval_step += self.cfg.eval_frequency

            if done:
                if self.step > 0:
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps), ty='train')
                
                obs, _ = self.env.reset(); done = False; episode_reward = 0; episode += 1; start_time = time.time()
            
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent): action = self.agent.act(obs, sample=True)
            
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            
            next_obs, reward, term, trunc, _ = self.env.step(action); done = term or trunc
            episode_reward += reward
            
            action_to_store = np.array([action]) if isinstance(self.env.action_space, gym.spaces.Discrete) else action
            self.replay_buffer.add(obs, action_to_store, reward, next_obs, float(done), 0.0)
            obs, self.step = next_obs, self.step + 1

@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    workspace = Workspace(cfg); workspace.run()

if __name__ == '__main__':
    main()
