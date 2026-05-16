import torch
import torch.nn as nn
import utils

class DiscreteDoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.Q1 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.Q2 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.Q1(obs), self.Q2(obs)
