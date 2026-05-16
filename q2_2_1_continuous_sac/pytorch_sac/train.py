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

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]

        self.agent = SACAgent(
            obs_dim=cfg.agent.params.obs_dim,
            action_dim=cfg.agent.params.action_dim,
            action_range=cfg.agent.params.action_range,
            device=self.device,
            critic_cfg=cfg.agent.params.critic,
            actor_cfg=cfg.agent.params.actor,
            discount=cfg.agent.params.discount,
            init_temperature=cfg.agent.params.init_temperature,
            alpha_lr=cfg.agent.params.alpha_lr,
            alpha_betas=cfg.agent.params.alpha_betas,
            actor_lr=cfg.agent.params.actor_lr,
            actor_betas=cfg.agent.params.actor_betas,
            actor_update_frequency=cfg.agent.params.actor_update_frequency,
            critic_lr=cfg.agent.params.critic_lr,
            critic_betas=cfg.agent.params.critic_betas,
            critic_tau=cfg.agent.params.critic_tau,
            critic_target_update_frequency=cfg.agent.params.critic_target_update_frequency,
            batch_size=cfg.agent.params.batch_size,
            learnable_temperature=cfg.agent.params.learnable_temperature
        )

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape, int(cfg.replay_buffer_capacity), self.device)
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0
        self._next_eval_step = 0

    def evaluate(self):
        print(f"\n>>> Running Evaluation at Step {self.step}")
        avg_reward = 0
        for _ in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            avg_reward += episode_reward
        
        final_eval = avg_reward / self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', final_eval, self.step)
        self.logger.dump(self.step, ty='eval')
        print(f">>> Evaluation Finished. Reward: {final_eval:.2f}\n")

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

                obs, _ = self.env.reset()
                done, episode_reward = False, 0
                episode += 1
                start_time = time.time()

            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            
            done_bool = float(done)
            done_no_max = 0 if trunc and not term else done_bool
            
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done_bool, done_no_max)
            obs = next_obs
            self.step += 1

@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
