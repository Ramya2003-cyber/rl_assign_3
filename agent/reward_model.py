from __future__ import annotations

import collections
import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """Reward network r_θ(s,a) → ℝ trained with Bradley-Terry preference loss."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cpu",
        val_fraction: float = 0.1,
    ):
        super().__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = torch.device(device)
        self.val_fraction = val_fraction

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted reward scalar for each (obs, action) pair."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    def segment_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Sum r_θ over a trajectory segment; returns (B, 1)."""
        B, T, _ = states.shape
        s_flat  = states.view(B * T, -1)
        a_flat  = actions.view(B * T, -1)
        r_flat  = self.forward(s_flat, a_flat)          # (B*T, 1)
        r_seg   = r_flat.view(B, T, 1).sum(dim=1)       # (B, 1)
        return r_seg

    def _bt_loss(
        self,
        states_1: torch.Tensor, actions_1: torch.Tensor,
        states_2: torch.Tensor, actions_2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Bradley-Terry cross-entropy loss; labels: 1=σ1 preferred, 0=σ2, 0.5=tie."""
        R1 = self.segment_return(states_1, actions_1)
        R2 = self.segment_return(states_2, actions_2)
        # log P(σ1 ≻ σ2) = log sigmoid(R1 - R2)
        logits = R1 - R2
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    @torch.no_grad()
    def _accuracy(
        self,
        states_1: torch.Tensor, actions_1: torch.Tensor,
        states_2: torch.Tensor, actions_2: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Fraction of preference pairs correctly predicted (ties excluded)."""
        R1 = self.segment_return(states_1, actions_1)
        R2 = self.segment_return(states_2, actions_2)
        pred_pref = (R1 > R2).float()   # 1 if model predicts σ1 preferred

        # Only evaluate on non-tie samples (label != 0.5)
        non_tie_mask = (labels != 0.5).squeeze(-1)
        if non_tie_mask.sum() == 0:
            return float("nan")

        correct = (pred_pref.squeeze(-1)[non_tie_mask] == labels.squeeze(-1)[non_tie_mask]).float()
        return correct.mean().item()

    def fit(
        self,
        dataset: "PreferenceDataset",
        num_epochs: int = 10,
        batch_size: int = 64,
    ) -> dict:
        """Train reward model; returns history dict with train/val loss and accuracy."""
        n = len(dataset)
        if n < 2:
            return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Train / val split
        indices     = list(range(n))
        random.shuffle(indices)
        n_val       = max(1, int(n * self.val_fraction))
        val_idx     = indices[:n_val]
        train_idx   = indices[n_val:]

        def _batch_generator(idxs):
            random.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                batch_idxs = idxs[start : start + batch_size]
                yield dataset.get_batch(batch_idxs, device=self.device)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(num_epochs):
            self.train()
            epoch_losses, epoch_accs = [], []
            for s1, a1, s2, a2, lbl in _batch_generator(train_idx):
                loss = self._bt_loss(s1, a1, s2, a2, lbl)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                epoch_accs.append(self._accuracy(s1, a1, s2, a2, lbl))

            history["train_loss"].append(float(np.mean(epoch_losses)))
            valid_accs = [x for x in epoch_accs if not np.isnan(x)]
            history["train_acc"].append(float(np.mean(valid_accs)) if valid_accs else float("nan"))

            self.eval()
            val_losses, val_accs = [], []
            with torch.no_grad():
                for s1, a1, s2, a2, lbl in _batch_generator(val_idx):
                    val_losses.append(self._bt_loss(s1, a1, s2, a2, lbl).item())
                    val_accs.append(self._accuracy(s1, a1, s2, a2, lbl))

            history["val_loss"].append(float(np.mean(val_losses)))
            valid_val_accs = [x for x in val_accs if not np.isnan(x)]
            history["val_acc"].append(float(np.mean(valid_val_accs)) if valid_val_accs else float("nan"))

        return history

    @torch.no_grad()
    def relabel_buffer(self, replay_buffer: "PEBBLEReplayBuffer", batch_size: int = 1024):
        """Overwrite replay_buffer.rewards with r_θ(s,a) for every stored transition."""
        self.eval()
        n = replay_buffer.capacity if replay_buffer.full else replay_buffer.idx
        dev = self.device

        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            obs  = torch.as_tensor(replay_buffer.obses[start:end],  device=dev).float()
            acts = torch.as_tensor(replay_buffer.actions[start:end], device=dev).float()
            new_r = self.forward(obs, acts).cpu().numpy()             # (chunk, 1)
            replay_buffer.rewards[start:end] = new_r

        self.train()


class PreferenceDataset:
    """Circular buffer of (σ1, σ2, label) preference triples."""

    def __init__(self, capacity: int = 5_000):
        self.capacity = capacity
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def add(
        self,
        seg1_states:  np.ndarray,
        seg1_actions: np.ndarray,
        seg2_states:  np.ndarray,
        seg2_actions: np.ndarray,
        label: float,
    ):
        self._buf.append({
            "s1": seg1_states.astype(np.float32),
            "a1": seg1_actions.astype(np.float32),
            "s2": seg2_states.astype(np.float32),
            "a2": seg2_actions.astype(np.float32),
            "y":  float(label),
        })

    def __len__(self):
        return len(self._buf)

    def get_batch(self, indices: list[int], device="cpu"):
        """Return (s1, a1, s2, a2, labels) tensors for given indices."""
        items = [self._buf[i] for i in indices]
        s1 = torch.tensor(np.stack([x["s1"] for x in items]), dtype=torch.float32, device=device)
        a1 = torch.tensor(np.stack([x["a1"] for x in items]), dtype=torch.float32, device=device)
        s2 = torch.tensor(np.stack([x["s2"] for x in items]), dtype=torch.float32, device=device)
        a2 = torch.tensor(np.stack([x["a2"] for x in items]), dtype=torch.float32, device=device)
        y  = torch.tensor([[x["y"]] for x in items], dtype=torch.float32, device=device)
        return s1, a1, s2, a2, y


class SimulatedTeacher:
    """Oracle that labels segment pairs using ground-truth reward sums (1/0/0.5)."""

    def __init__(
        self,
        reward_type: Literal["a", "b", "c"],
        max_feedback: int = 2_000,
        equal_threshold: float = 1e-4,
    ):
        assert reward_type in ("a", "b", "c")
        self.reward_type     = reward_type
        self.max_feedback    = max_feedback
        self.equal_threshold = equal_threshold
        self.queries_used    = 0

    @property
    def budget_remaining(self) -> int:
        return self.max_feedback - self.queries_used

    @property
    def budget_exhausted(self) -> bool:
        return self.queries_used >= self.max_feedback

    def _gt_reward_step(self, state, action, reward_type):
        """Not implemented; use query() with segments carrying gt_rewards."""
        raise NotImplementedError(
            "Direct re-derivation is not implemented; use query() with "
            "segments that include the 'gt_rewards' array."
        )

    def query(
        self,
        seg1_gt_rewards: np.ndarray,
        seg2_gt_rewards: np.ndarray,
    ) -> float | None:
        """Label a segment pair; returns 1.0/0.0/0.5, or None if budget exhausted."""
        if self.budget_exhausted:
            return None

        R1 = float(np.sum(seg1_gt_rewards))
        R2 = float(np.sum(seg2_gt_rewards))
        self.queries_used += 1

        if abs(R1 - R2) <= self.equal_threshold:
            return 0.5
        return 1.0 if R1 > R2 else 0.0


class PEBBLEReplayBuffer:
    """Replay buffer extended with gt_rewards for PEBBLE preference labelling."""

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        capacity: int,
        device: str,
    ):
        self.capacity = capacity
        self.device   = torch.device(device)

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses          = np.empty((capacity, *obs_shape),    dtype=obs_dtype)
        self.next_obses     = np.empty((capacity, *obs_shape),    dtype=obs_dtype)
        self.actions        = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards        = np.empty((capacity, 1),             dtype=np.float32)
        self.gt_rewards     = np.empty((capacity, 1),             dtype=np.float32)
        self.not_dones      = np.empty((capacity, 1),             dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1),           dtype=np.float32)

        self.idx  = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx],            obs)
        np.copyto(self.actions[self.idx],          action)
        np.copyto(self.rewards[self.idx],          reward)
        np.copyto(self.gt_rewards[self.idx],       reward)   # keep a clean copy
        np.copyto(self.next_obses[self.idx],       next_obs)
        np.copyto(self.not_dones[self.idx],        not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx  = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        """Sample a random mini-batch.  Returns tensors on ``self.device``."""
        n    = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, n, size=batch_size)

        obses  = torch.as_tensor(self.obses[idxs],       device=self.device).float()
        acts   = torch.as_tensor(self.actions[idxs],     device=self.device)
        rews   = torch.as_tensor(self.rewards[idxs],     device=self.device)
        n_obs  = torch.as_tensor(self.next_obses[idxs],  device=self.device).float()
        nd     = torch.as_tensor(self.not_dones[idxs],   device=self.device)
        nd_nm  = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, acts, rews, n_obs, nd, nd_nm

    def sample_segment(self, seg_len: int) -> dict:
        """Sample a contiguous segment; returns dict with states/actions/gt_rewards."""
        n = self.capacity if self.full else self.idx
        if n < seg_len:
            raise ValueError(f"Buffer only has {n} transitions, need at least {seg_len}.")

        if self.full:
            start = np.random.randint(0, self.capacity - seg_len + 1)
        else:
            start = np.random.randint(0, n - seg_len + 1)

        sl = slice(start, start + seg_len)
        return {
            "states":     self.obses[sl].copy(),
            "actions":    self.actions[sl].copy(),
            "gt_rewards": self.gt_rewards[sl, 0].copy(),
        }
