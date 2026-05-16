import numpy as np
import torch
import torch.nn.functional as F
from agent import Agent
import utils
from agent.discrete_actor import DiscreteActor
from agent.discrete_critic import DiscreteDoubleQCritic

class DiscreteSACAgent(Agent):
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        super().__init__()
        self.device = torch.device(device)
        self.discount, self.critic_tau = discount, critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size, self.learnable_temperature = batch_size, learnable_temperature

        self.critic = DiscreteDoubleQCritic(obs_dim, action_dim, critic_cfg['hidden_dim'], critic_cfg['hidden_depth']).to(self.device)
        self.critic_target = DiscreteDoubleQCritic(obs_dim, action_dim, critic_cfg['hidden_dim'], critic_cfg['hidden_depth']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DiscreteActor(obs_dim, action_dim, actor_cfg['hidden_dim'], actor_cfg['hidden_depth']).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device).requires_grad_(True)
        self.target_entropy = 0.2 * (-np.log(1.0 / action_dim))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self): return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = self.actor.act(obs, sample)
        return action.item()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, _ = replay_buffer.sample(self.batch_size)
        action = action.long() 

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            t_Q1, t_Q2 = self.critic_target(next_obs)
            target_V = (next_probs * (torch.min(t_Q1, t_Q2) - self.alpha * next_log_probs)).sum(-1, keepdim=True)
            target_Q = reward + (not_done * self.discount * target_V)

        curr_Q1, curr_Q2 = self.critic(obs)
        curr_Q1 = curr_Q1.gather(1, action)
        curr_Q2 = curr_Q2.gather(1, action)
        
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()

        if step % self.actor_update_frequency == 0:
            probs, log_probs = self.actor(obs)
            Q1, Q2 = self.critic(obs)
            Q = torch.min(Q1, Q2)
            
            actor_loss = (probs * (self.alpha.detach() * log_probs - Q)).sum(-1).mean()
            logger.log('train_actor/loss', actor_loss, step)
            
            entropy = -(probs * log_probs).sum(-1).mean()
            logger.log('train_actor/entropy', entropy, step)

            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

            if self.learnable_temperature:
                alpha_loss = (probs.detach() * (-self.alpha * (log_probs + self.target_entropy).detach())).sum(-1).mean()
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
                
                self.log_alpha_optimizer.zero_grad(); alpha_loss.backward(); self.log_alpha_optimizer.step()
        
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
