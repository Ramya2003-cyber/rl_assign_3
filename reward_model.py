"""
reward_model.py
===============
PEBBLE reward-learning components for dm_control Reacher.

Components
----------
RewardModel      – Neural network r_θ(s,a) trained with Bradley-Terry loss.
                   Tracks a hold-out validation split and logs train/val accuracy.
SimulatedTeacher – Oracle that labels segment pairs with the ground-truth reward
                   of a chosen formulation ('a', 'b', or 'c').
PreferenceDataset – Fixed-capacity circular buffer for (σ1, σ2, label) triples.
"""

from __future__ import annotations

import collections
import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Reward network  r_θ : (s, a) → ℝ.

    Architecture
    ------------
    Two hidden layers of size ``hidden_dim`` with ReLU activations.

    Training
    --------
    Uses the Bradley-Terry cross-entropy preference loss:

        L = -E[ y·log P(σ1≻σ2) + (1-y)·log P(σ2≻σ1) ]

    where y=1 means σ1 is preferred, y=0 means σ2 is preferred,
    and y=0.5 encodes a tie.

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_dim : int
    lr : float        Adam learning rate.
    device : str
    val_fraction : float
        Fraction of preference pairs held out for validation (default 10 %).
    """

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

        # ── Network ──────────────────────────────────────────────────────────
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

        # ── Optimiser ────────────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return predicted reward scalar for each (obs, action) pair.

        Args
        ----
        obs    : (B, obs_dim)
        action : (B, action_dim)

        Returns
        -------
        Tensor of shape (B, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    # ── Segment-level utility ─────────────────────────────────────────────────

    def segment_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of predicted rewards over a trajectory segment.

        Args
        ----
        states  : (B, T, obs_dim)
        actions : (B, T, action_dim)

        Returns
        -------
        Tensor of shape (B, 1).
        """
        B, T, _ = states.shape
        # Flatten to (B*T, dim), run network, reshape and sum over T.
        s_flat  = states.view(B * T, -1)
        a_flat  = actions.view(B * T, -1)
        r_flat  = self.forward(s_flat, a_flat)          # (B*T, 1)
        r_seg   = r_flat.view(B, T, 1).sum(dim=1)       # (B, 1)
        return r_seg

    # ── Bradley-Terry preference loss ────────────────────────────────────────

    def _bt_loss(
        self,
        states_1: torch.Tensor, actions_1: torch.Tensor,
        states_2: torch.Tensor, actions_2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley-Terry cross-entropy loss.

        labels : (B, 1)  – 1 → σ1 preferred, 0 → σ2 preferred, 0.5 → tie.

        For ties we average the two possible cross-entropy terms.
        """
        R1 = self.segment_return(states_1, actions_1)   # (B, 1)
        R2 = self.segment_return(states_2, actions_2)   # (B, 1)

        # log P(σ1 ≻ σ2) = log sigmoid(R1 - R2)
        logits = R1 - R2        # (B, 1) – positive → σ1 preferred

        # BCEWithLogitsLoss handles the sigmoid + log numerically stably
        # and supports soft labels (0.5 for tie).
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    # ── Accuracy helper ───────────────────────────────────────────────────────

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

    # ── Training entry-point ──────────────────────────────────────────────────

    def fit(
        self,
        dataset: "PreferenceDataset",
        num_epochs: int = 10,
        batch_size: int = 64,
    ) -> dict:
        """
        Train the reward model on *dataset* for ``num_epochs`` epochs.

        Internally creates a train/val split (``val_fraction``).

        Returns
        -------
        dict with keys:
            'train_loss'    : list of per-epoch average training losses
            'val_loss'      : list of per-epoch validation losses
            'train_acc'     : list of per-epoch training accuracies
            'val_acc'       : list of per-epoch validation accuracies
        """
        n = len(dataset)
        if n < 2:
            return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Train / val split (stratified by order in buffer, simple approach)
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
            # ── Train ──────────────────────────────────────────────────────
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

            # ── Validation ────────────────────────────────────────────────
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

    # ── Buffer relabelling ────────────────────────────────────────────────────

    @torch.no_grad()
    def relabel_buffer(self, replay_buffer: "PEBBLEReplayBuffer", batch_size: int = 1024):
        """
        Overwrite ``replay_buffer.rewards`` with r_θ(s,a) for every stored
        transition.  This is the critical off-policy relabelling step from the
        PEBBLE paper.

        Parameters
        ----------
        replay_buffer : PEBBLEReplayBuffer
            Must expose ``.obses``, ``.actions``, ``.rewards``, ``.idx``,
            ``.full``, ``.capacity``.
        batch_size : int
            Number of transitions to process in one GPU batch to avoid OOM.
        """
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


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------

class PreferenceDataset:
    """
    Circular buffer of (σ1, σ2, label) preference triples.

    Each segment σ is stored as a dict with keys ``'states'`` and
    ``'actions'``, both numpy arrays of shape (T, dim).

    Parameters
    ----------
    capacity : int   Maximum number of triples to keep.
    """

    def __init__(self, capacity: int = 5_000):
        self.capacity = capacity
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    # ── Insertion ─────────────────────────────────────────────────────────────

    def add(
        self,
        seg1_states:  np.ndarray,   # (T, obs_dim)
        seg1_actions: np.ndarray,   # (T, action_dim)
        seg2_states:  np.ndarray,
        seg2_actions: np.ndarray,
        label: float,               # 1 / 0 / 0.5
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

    # ── Batch access ──────────────────────────────────────────────────────────

    def get_batch(self, indices: list[int], device="cpu"):
        """
        Gather a batch of preference triples by index.

        Returns
        -------
        (s1, a1, s2, a2, labels)  – all torch.Tensor on ``device``.
        s1 : (B, T, obs_dim)
        a1 : (B, T, action_dim)
        ...
        labels : (B, 1)
        """
        items = [self._buf[i] for i in indices]
        s1 = torch.tensor(np.stack([x["s1"] for x in items]), dtype=torch.float32, device=device)
        a1 = torch.tensor(np.stack([x["a1"] for x in items]), dtype=torch.float32, device=device)
        s2 = torch.tensor(np.stack([x["s2"] for x in items]), dtype=torch.float32, device=device)
        a2 = torch.tensor(np.stack([x["a2"] for x in items]), dtype=torch.float32, device=device)
        y  = torch.tensor([[x["y"]] for x in items], dtype=torch.float32, device=device)
        return s1, a1, s2, a2, y


# ---------------------------------------------------------------------------
# SimulatedTeacher
# ---------------------------------------------------------------------------

class SimulatedTeacher:
    """
    Oracle teacher that evaluates segment pairs with the *ground-truth* reward
    of a chosen formulation.

    The teacher computes:

        R_gt(σ) = Σ_{t ∈ σ} r_gt(s_t, a_t)

    and returns a label:

        y = 1    if R_gt(σ1) > R_gt(σ2)
        y = 0    if R_gt(σ2) > R_gt(σ1)
        y = 0.5  if equal (tie)

    Parameters
    ----------
    reward_type : str        One of 'a', 'b', 'c'.
    max_feedback : int       Maximum number of queries the teacher will answer.
                             Once exhausted, ``query`` returns None.
    equal_threshold : float  Values within this threshold are treated as a tie.
    """

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

    # ── Ground-truth reward for a single (s, a) transition ───────────────────

    def _gt_reward_step(
        self,
        state:  np.ndarray,   # (obs_dim,)
        action: np.ndarray,   # (action_dim,)
        reward_type: str,
    ) -> float:
        """
        Compute the ground-truth scalar reward for one transition.

        The raw environment reward is carried along inside each transition
        stored in the ``PEBBLEReplayBuffer`` (``gt_rewards`` array).  This
        function is a *fallback* that re-derives the reward from the stored
        observation.  However, because Reacher's physics is embedded inside
        dm_control, re-derivation is non-trivial, so we instead store the
        ground-truth reward alongside every transition and read it back.

        In practice ``query`` is always called with segments that include the
        pre-stored ``gt_rewards``.  This function is kept for completeness /
        testing.
        """
        raise NotImplementedError(
            "Direct re-derivation is not implemented; use query() with "
            "segments that include the 'gt_rewards' array."
        )

    # ── Query interface ───────────────────────────────────────────────────────

    def query(
        self,
        seg1_gt_rewards: np.ndarray,  # (T,) ground-truth rewards for segment 1
        seg2_gt_rewards: np.ndarray,  # (T,) ground-truth rewards for segment 2
    ) -> float | None:
        """
        Compare two segments using their ground-truth reward sums.

        Parameters
        ----------
        seg1_gt_rewards, seg2_gt_rewards : np.ndarray, shape (T,)
            Pre-computed ground-truth rewards for each step in the segments.

        Returns
        -------
        float  – 1.0 (σ1 preferred), 0.0 (σ2 preferred), or 0.5 (tie).
        None   – budget exhausted, query rejected.
        """
        if self.budget_exhausted:
            return None

        R1 = float(np.sum(seg1_gt_rewards))
        R2 = float(np.sum(seg2_gt_rewards))
        self.queries_used += 1

        if abs(R1 - R2) <= self.equal_threshold:
            return 0.5
        return 1.0 if R1 > R2 else 0.0


# ---------------------------------------------------------------------------
# PEBBLEReplayBuffer
# ---------------------------------------------------------------------------

class PEBBLEReplayBuffer:
    """
    Extended replay buffer that stores ground-truth rewards alongside the
    reward-model rewards.

    Extra arrays vs. the base ``ReplayBuffer``
    -------------------------------------------
    ``gt_rewards``   – raw ground-truth reward received from the environment.
                       Used by the teacher to label segments.
    ``rewards``      – the reward actually fed to SAC.  After relabelling this
                       stores r_θ(s,a); during seed-phase it stores gt_rewards.

    The buffer tracks write positions in a circular (overwrite) fashion.

    Parameters
    ----------
    obs_shape    : tuple
    action_shape : tuple
    capacity     : int
    device       : str  (torch device string)
    """

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

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,        # ground-truth reward (from the env)
        next_obs: np.ndarray,
        done: bool,
        done_no_max: bool,
    ):
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

    # ── Segment sampling for the teacher ─────────────────────────────────────

    def sample_segment(self, seg_len: int) -> dict:
        """
        Sample one *contiguous* segment of length ``seg_len`` from valid
        transitions.

        Returns a dict with keys:
            'states'     : np.ndarray (T, obs_dim)
            'actions'    : np.ndarray (T, action_dim)
            'gt_rewards' : np.ndarray (T,)   ← used by the teacher
        """
        n = self.capacity if self.full else self.idx
        if n < seg_len:
            raise ValueError(
                f"Buffer only has {n} transitions, need at least {seg_len}."
            )

        # Avoid wrapping across the boundary for simplicity
        # (wrap is rare once the buffer is well-filled)
        if self.full:
            # The "valid non-wrapping" range: [idx, idx + n - seg_len]
            # We can pick any start in [0, capacity - seg_len] when full,
            # but must avoid the write-cursor neighbourhood.
            # Simple: sample from [0, n - seg_len] and shift by idx so we
            # pick a uniformly random contiguous window.
            start = np.random.randint(0, self.capacity - seg_len + 1)
        else:
            start = np.random.randint(0, n - seg_len + 1)

        sl = slice(start, start + seg_len)
        return {
            "states":     self.obses[sl].copy(),
            "actions":    self.actions[sl].copy(),
            "gt_rewards": self.gt_rewards[sl, 0].copy(),
        }
