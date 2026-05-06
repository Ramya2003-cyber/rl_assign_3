#!/usr/bin/env python3
"""
run_reacher.py
==============
Training loop for Question 2.3: SAC on dm_control reacher-easy with three
reward formulations (Ra, Rb, Rc) and cross-evaluation.

Usage (Hydra):
    python run_reacher.py reward_type=a seed=1
    python run_reacher.py reward_type=b seed=1
    python run_reacher.py reward_type=c seed=1

The script saves a .npy results file structured for easy multi-seed plotting:
    reacher_results_<reward_type>_seed_<seed>.npy

Shape of the saved array: (num_eval_checkpoints, 3, num_eval_episodes)
    axis-0 : evaluation checkpoint index  (every 10_000 training steps)
    axis-1 : reward type index            (0→Ra, 1→Rb, 2→Rc)
    axis-2 : episode index                (0..19)

To compute mean / std across seeds, load all seeds and stack along axis-0.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_buffer import ReplayBuffer
from agent.sac import SACAgent
import utils

from envs.reacher_custom import ReacherWrapper


# ---------------------------------------------------------------------------
# Dummy Logger (same pattern used elsewhere in this codebase)
# ---------------------------------------------------------------------------

class DummyLogger:
    """Minimal logger that swallows all calls silently."""
    def log(self, *args, **kwargs): pass
    def log_histogram(self, *args, **kwargs): pass
    def log_param(self, *args, **kwargs): pass
    def dump(self, *args, **kwargs): pass


# ---------------------------------------------------------------------------
# Cross-evaluation
# ---------------------------------------------------------------------------

def evaluate_all_rewards(
    agent: SACAgent,
    reward_type: str,
    num_eval_episodes: int = 20,
    rc_step_limit: int = 1000,
    rc_timeout_penalty: float = -20.0,
) -> dict[str, list[float]]:
    """
    Run ``num_eval_episodes`` episodes with the deterministic policy on the
    TRAINING reward formulation's environment, but record returns for all three
    reward formulations simultaneously.

    For Rc evaluation episodes:
      - We do NOT allow the episode to run indefinitely.
      - If an episode hits ``rc_step_limit`` steps without termination we
        truncate it, take the base return of ``-rc_step_limit`` and subtract the
        timeout penalty, logging ``-(rc_step_limit + |rc_timeout_penalty|)``.
      - (The -1000 base return is already accumulated in ep_returns['c'] by then.)

    Parameters
    ----------
    agent : SACAgent
        The agent to evaluate (will be placed in eval mode automatically).
    reward_type : str
        The reward type that the training environment uses (determines which
        environment to run the policy on).  All three reward returns are still
        computed at every step.
    num_eval_episodes : int
        Number of evaluation episodes.
    rc_step_limit : int
        Maximum steps before an Rc evaluation episode is forcibly truncated.
    rc_timeout_penalty : float
        Extra penalty added to the Rc return when the evaluation episode is
        truncated (should be negative, e.g. -20.0).

    Returns
    -------
    dict mapping 'a', 'b', 'c' → list of per-episode returns (length = num_eval_episodes)
    """
    # Evaluation uses the same reward-type environment (so the policy sees a
    # consistent observation).  But we track all three returns via
    # `compute_all_rewards`.
    eval_env = ReacherWrapper(reward_type=reward_type, seed=0)

    ep_returns: dict[str, list[float]] = {'a': [], 'b': [], 'c': []}

    agent.train(False)  # deterministic eval mode

    for ep in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        sums = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        step_in_ep = 0
        rc_truncated = False

        while not done:
            with torch.no_grad():
                action = agent.act(obs, sample=False)

            obs, _, terminated, truncated, info = eval_env.step(action)
            all_r = info.get('all_rewards', eval_env.compute_all_rewards(action))

            sums['a'] += all_r['a']
            sums['b'] += all_r['b']
            sums['c'] += all_r['c']

            step_in_ep += 1
            done = terminated or truncated

            # Rc evaluation cap: truncate at rc_step_limit
            if step_in_ep >= rc_step_limit and not done:
                rc_truncated = True
                done = True

        # Apply the Rc evaluation timeout penalty if we truncated
        if rc_truncated:
            sums['c'] += rc_timeout_penalty

        for k in ('a', 'b', 'c'):
            ep_returns[k].append(sums[k])

    eval_env.close()
    agent.train(True)  # restore training mode
    return ep_returns


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(cfg: DictConfig):
    """Main training entry-point (called by Hydra)."""

    reward_type: str = cfg.reward_type          # 'a', 'b', or 'c'
    seed: int        = int(cfg.seed)
    device_str: str  = cfg.get('device', 'cpu')
    num_train_steps: int = int(cfg.get('num_train_steps', 500_000))
    num_seed_steps:  int = int(cfg.get('num_seed_steps',   10_000))
    eval_frequency:  int = int(cfg.get('eval_frequency',   10_000))
    num_eval_episodes: int = int(cfg.get('num_eval_episodes', 20))
    replay_buffer_capacity: int = int(cfg.get('replay_buffer_capacity', 1_000_000))

    print(f"\n{'='*60}")
    print(f"  Reacher SAC  |  reward_type={reward_type}  |  seed={seed}")
    print(f"  device={device_str}  |  total_steps={num_train_steps}")
    print(f"{'='*60}\n")

    # ── Seeding ─────────────────────────────────────────────────────────────
    utils.set_seed_everywhere(seed)
    device = torch.device(device_str)

    # ── Environment ──────────────────────────────────────────────────────────
    env = ReacherWrapper(reward_type=reward_type, seed=seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  action_range={action_range}\n")

    # ── Agent ────────────────────────────────────────────────────────────────
    # critic_cfg = hydra.utils.instantiate(
    #     cfg.agent.critic_cfg,
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     _recursive_=False,
    # )
    # actor_cfg = hydra.utils.instantiate(
    #     cfg.agent.actor_cfg,
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     _recursive_=False,
    # )

    # agent = SACAgent(
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     action_range=action_range,
    #     device=device_str,
    #     critic_cfg=cfg.agent.critic_cfg,
    #     actor_cfg=cfg.agent.actor_cfg,
    #     discount=float(cfg.agent.get('discount', 0.99)),
    #     init_temperature=float(cfg.agent.get('init_temperature', 0.1)),
    #     alpha_lr=float(cfg.agent.get('alpha_lr', 1e-4)),
    #     alpha_betas=list(cfg.agent.get('alpha_betas', [0.9, 0.999])),
    #     actor_lr=float(cfg.agent.get('actor_lr', 1e-4)),
    #     actor_betas=list(cfg.agent.get('actor_betas', [0.9, 0.999])),
    #     actor_update_frequency=int(cfg.agent.get('actor_update_frequency', 1)),
    #     critic_lr=float(cfg.agent.get('critic_lr', 1e-4)),
    #     critic_betas=list(cfg.agent.get('critic_betas', [0.9, 0.999])),
    #     critic_tau=float(cfg.agent.get('critic_tau', 0.005)),
    #     critic_target_update_frequency=int(cfg.agent.get('critic_target_update_frequency', 2)),
    #     batch_size=int(cfg.agent.get('batch_size', 1024)),
    #     learnable_temperature=bool(cfg.agent.get('learnable_temperature', True)),
    # )
    # ── Agent ────────────────────────────────────────────────────────────────
    # Temporarily disable Hydra's strict struct mode so we can inject variables
    OmegaConf.set_struct(cfg, False) 
    
    # 1. Inject dimensions into the main agent config
    cfg.agent.obs_dim = obs_dim
    cfg.agent.action_dim = action_dim
    cfg.agent.action_range = action_range
    
    # 2. Force inject into the nested actor/critic configs to prevent missing value errors
    if 'critic_cfg' in cfg.agent:
        cfg.agent.critic_cfg.obs_dim = obs_dim
        cfg.agent.critic_cfg.action_dim = action_dim
    if 'actor_cfg' in cfg.agent:
        cfg.agent.actor_cfg.obs_dim = obs_dim
        cfg.agent.actor_cfg.action_dim = action_dim

    # 3. Cleanly instantiate the agent exactly how pytorch_sac expects
    agent = hydra.utils.instantiate(cfg.agent)

    # ── Replay buffer ────────────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=replay_buffer_capacity,
        device=device,
    )

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = DummyLogger()

    # ── Results storage ──────────────────────────────────────────────────────
    # We accumulate one row per evaluation checkpoint.
    # Each row: shape (3, num_eval_episodes)  →  rows for Ra, Rb, Rc.
    eval_checkpoints: list[np.ndarray] = []   # list of (3, E) arrays
    checkpoint_steps: list[int] = []          # global step at each checkpoint

    # ── Helper: run evaluation and store results ─────────────────────────────
    def _run_eval(global_step: int):
        print(f"\n  [Eval @ step {global_step}] running {num_eval_episodes} episodes …")
        t0 = time.time()
        ep_returns = evaluate_all_rewards(
            agent=agent,
            reward_type=reward_type,
            num_eval_episodes=num_eval_episodes,
        )
        row = np.stack([
            np.array(ep_returns['a']),
            np.array(ep_returns['b']),
            np.array(ep_returns['c']),
        ])  # shape (3, E)
        eval_checkpoints.append(row)
        checkpoint_steps.append(global_step)

        print(f"  Ra mean={np.mean(ep_returns['a']):.3f} ± {np.std(ep_returns['a']):.3f}")
        print(f"  Rb mean={np.mean(ep_returns['b']):.3f} ± {np.std(ep_returns['b']):.3f}")
        print(f"  Rc mean={np.mean(ep_returns['c']):.3f} ± {np.std(ep_returns['c']):.3f}")
        print(f"  (eval took {time.time()-t0:.1f}s)")

    # ── BASELINE EVAL at step 0 (random-init policy) ────────────────────────
    _run_eval(global_step=0)

    # ── Training ─────────────────────────────────────────────────────────────
    global_step   = 0
    episode       = 0
    next_eval_at  = eval_frequency          # first eval checkpoint after step 0

    obs, _ = env.reset()
    ep_reward     = 0.0
    ep_step       = 0
    ep_start_time = time.time()

    while global_step < num_train_steps:

        # ── Select action ────────────────────────────────────────────────────
        if global_step < num_seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # ── Environment step ─────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)

        # "done" flag for the critic's bootstrap: True only on true termination.
        # We do NOT bootstrap through a timeout boundary (done_no_max = 0 there).
        episode_ended    = terminated or truncated
        done_for_buffer  = float(terminated)        # genuine terminal state
        done_no_max      = float(terminated)        # same — no max-truncation bootstrap

        replay_buffer.add(obs, action, reward, next_obs, done_for_buffer, done_no_max)

        ep_reward += reward
        ep_step   += 1
        global_step += 1

        # ── Update agent (after seed phase) ──────────────────────────────────
        if global_step >= num_seed_steps and len(replay_buffer) >= agent.batch_size:
            agent.update(replay_buffer, logger, global_step)

        # ── Periodic evaluation ──────────────────────────────────────────────
        if global_step >= next_eval_at:
            _run_eval(global_step=global_step)
            next_eval_at += eval_frequency

        # ── Episode bookkeeping ──────────────────────────────────────────────
        if episode_ended:
            print(
                f"  Ep {episode:5d}  |  step {global_step:7d}  |  "
                f"ep_steps {ep_step:5d}  |  ep_reward {ep_reward:.2f}  |  "
                f"time {time.time()-ep_start_time:.1f}s"
            )
            episode += 1
            ep_reward = 0.0
            ep_step   = 0
            ep_start_time = time.time()
            obs, _ = env.reset()
        else:
            obs = next_obs

    # ── Final eval (if the last checkpoint hasn't been hit yet) ──────────────
    if checkpoint_steps[-1] < global_step:
        _run_eval(global_step=global_step)

    env.close()

    # ── Save results ──────────────────────────────────────────────────────────
    # Shape: (num_checkpoints, 3, num_eval_episodes)
    results_array = np.stack(eval_checkpoints, axis=0)
    steps_array   = np.array(checkpoint_steps)

    out_path = f"reacher_results_{reward_type}_seed_{seed}.npy"
    np.save(out_path, {
        'returns':          results_array,    # (C, 3, E)  — C checkpoints, 3 rewards, E episodes
        'checkpoint_steps': steps_array,      # (C,)
        'reward_type':      reward_type,
        'seed':             seed,
    })
    print(f"\n  Results saved → {out_path}")
    print(f"  Array shape: {results_array.shape}  (checkpoints, reward_types, episodes)")
    print("  Done.\n")


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path='config', config_name='reacher', version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
