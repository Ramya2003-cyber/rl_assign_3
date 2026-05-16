import collections
import random
import numpy as np
import torch

class PreferenceBuffer:
   
    def __init__(self, capacity=10000):
      
        self.capacity = capacity
        # Use a deque for FIFO behavior when the buffer is full
        self.buffer = collections.deque(maxlen=capacity)
        
    def store(self, segment_A, segment_B, label):
        
        self.buffer.append((segment_A, segment_B, label))
        
    def sample(self, batch_size, device='cpu'):
       
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states_A, actions_A, states_B, actions_B, labels = [], [], [], [], []
        
        for seg_A, seg_B, label in batch:
            sa, aa = seg_A
            sb, ab = seg_B
            
            states_A.append(sa)
            actions_A.append(aa)
            states_B.append(sb)
            actions_B.append(ab)
            labels.append([label])
            
        # Convert the lists of arrays into PyTorch tensors
        states_A_tensor = torch.tensor(np.array(states_A), dtype=torch.float32, device=device)
        actions_A_tensor = torch.tensor(np.array(actions_A), dtype=torch.float32, device=device)
        states_B_tensor = torch.tensor(np.array(states_B), dtype=torch.float32, device=device)
        actions_B_tensor = torch.tensor(np.array(actions_B), dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32, device=device)
        
        segment_A_batch = (states_A_tensor, actions_A_tensor)
        segment_B_batch = (states_B_tensor, actions_B_tensor)
        
        return segment_A_batch, segment_B_batch, labels_tensor

    def __len__(self):
        return len(self.buffer)
