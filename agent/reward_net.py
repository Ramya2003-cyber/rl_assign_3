import torch
import torch.nn as nn

class RewardNet(nn.Module):
    """
    The Reward Predictor network for the PEBBLE algorithm.
    Takes in concatenated (state, action) and outputs a scalar reward prediction.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # We use BCEWithLogitsLoss for the Bradley-Terry preference loss
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, state, action):
        """
        Outputs the predicted reward for a given state-action pair.
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def preference_loss(self, segment_A, segment_B, labels):
        """
        Calculates the Bradley-Terry preference loss between pairs of segments.
        
        Args:
            segment_A: Tuple of (states, actions) for trajectory A.
                       states shape: (batch_size, segment_length, state_dim)
                       actions shape: (batch_size, segment_length, action_dim)
            segment_B: Tuple of (states, actions) for trajectory B.
            labels: Tensor of shape (batch_size, 1). 
                    1 means A is preferred, 0 means B is preferred, 0.5 is a tie.
                    
        Returns:
            Scalar loss value.
        """
        states_A, actions_A = segment_A
        states_B, actions_B = segment_B
        
        # Predict step-wise rewards. Shape: (batch_size, segment_length, 1)
        rewards_A = self.forward(states_A, actions_A)
        rewards_B = self.forward(states_B, actions_B)
        
        # Compute the sum of predicted rewards for each segment
        # Shape: (batch_size, 1)
        sum_R_A = torch.sum(rewards_A, dim=1)
        sum_R_B = torch.sum(rewards_B, dim=1)
        
        # Bradley-Terry model logit is the difference in sum of rewards: (R_A - R_B)
        # P(A > B) = sigmoid(R_A - R_B)
        logits = sum_R_A - sum_R_B
        
        # Calculate loss using BCEWithLogitsLoss for numerical stability
        loss = self.loss_fn(logits, labels.float())
        
        return loss
