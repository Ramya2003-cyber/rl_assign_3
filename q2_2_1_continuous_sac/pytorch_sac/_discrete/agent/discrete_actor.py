import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs):
        logits = self.trunk(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def act(self, obs, sample=False):
        probs, _ = self.forward(obs)
        if sample:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action
