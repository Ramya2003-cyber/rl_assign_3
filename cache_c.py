#!/usr/bin/env python3
"""
eval_q3_reacher.py
==================
Evaluate three trained SAC policies (Ra, Rb, Rc) on the custom dm_control
Reacher environment over 500 episodes each.

Metrics collected per episode (episode length = 5000 steps):
  steps_to_goal   – first timestep the fingertip enters the target (< 0.05 m).
                    Recorded as 5000 if the target is never reached.
  steps_in_target – total number of timesteps the fingertip spent inside the
                    target over the full 5000-step episode.

Output files (one pair per policy):
  reacher-a_steps_to_goal.npy      shape (500,)
  reacher-a_steps_in_target.npy    shape (500,)
  reacher-b_steps_to_goal.npy      ...
  ...

Usage
-----
    python eval_q3_reacher.py

Before running, update POLICY_PATHS below to point to your saved .pt files.
Each .pt file should be a dict produced by save_agent() at the bottom of this
file (or from your training loop checkpoint).

Saving weights from your training loop
---------------------------------------
At the end of run_reacher.py (or inside _run_eval), call:

    save_agent(agent, f"reacher_{reward_type}_seed_{seed}_final.pt")

The loader in this script will pick it up automatically.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
# ── Make the project root importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.reacher_custom import ReacherWrapper, _compute_distance
import utils


# ===========================================================================
# ▶  CONFIGURE THESE BEFORE RUNNING
# ===========================================================================

NUM_EPISODES: int   = 500
MAX_STEPS:    int   = 5000   # CRITICAL: override from default 1000

# Paths to your saved .pt weight files.
# Each file is expected to be the dict produced by save_agent() below.
# Replace the placeholder strings with your actual file paths.
POLICY_PATHS: dict[str, str] = {
   
    'reacher-c': 'reacher_c_seed_42_final.pt'

}

# Seed used when creating the evaluation environment (use 0 for consistency)
EVAL_ENV_SEED: int = 0

# Device for inference
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================================================================
# ▶  DISTANCE / IN-TARGET HELPER
# ===========================================================================

# The ReacherWrapper stores the dm_control physics object as `_physics`.
# We use the module-level `_compute_distance` from reacher_custom.py which
# reads `geom_xpos['finger']` and `geom_xpos['target']` directly from physics.
# This is the identical check the environment itself uses internally.

TARGET_RADIUS: float = 0.05  # must match ReacherWrapper._DIST_THRESHOLD


def is_in_target(env: ReacherWrapper, info: dict) -> bool:
    """
    Return True if the fingertip is currently inside the target sphere.

    We rely on ``info['in_target']`` which the ReacherWrapper already computes
    at every step using its own ``_arm_in_target()`` method (threshold 0.05 m).
    This is the canonical in-target check — no observation parsing needed.
    """
    return bool(info.get('in_target', False))


def get_distance(env: ReacherWrapper) -> float:
    """
    Euclidean distance between fingertip and target in the 2-D task plane.
    Calls the module-level helper from reacher_custom.py which reads the
    live dm_control physics object.
    """
    return _compute_distance(env._physics)


# ===========================================================================
# ▶  AGENT WEIGHT SAVE / LOAD HELPERS
# ===========================================================================

def save_agent(agent, path: str):
    """
    Save a SACAgent's learnable parameters to a single .pt file.

    Call this at the end of your training run:
        save_agent(agent, f"reacher_{reward_type}_seed_{seed}_final.pt")
    """
    torch.save({
        'actor_state_dict':         agent.actor.state_dict(),
        'critic_state_dict':        agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'log_alpha':                agent.log_alpha.detach().cpu(),
    }, path)
    print(f"  ✅ Agent weights saved → {path}")


def load_agent_weights(agent, path: str):
    """
    Load weights previously saved with save_agent() into an existing SACAgent.
    Only the actor is strictly required for evaluation; the rest are loaded
    when available.
    """
    ckpt = torch.load(path, map_location=agent.device)

    agent.actor.load_state_dict(ckpt['actor_state_dict'])

    if 'critic_state_dict' in ckpt:
        agent.critic.load_state_dict(ckpt['critic_state_dict'])
    if 'critic_target_state_dict' in ckpt:
        agent.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
    if 'log_alpha' in ckpt:
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt['log_alpha'].to(agent.device))

    print(f"  ✅ Weights loaded from {path}")


# ===========================================================================
# ▶  AGENT CONSTRUCTION  (mirrors run_reacher.py exactly)
# ===========================================================================

class DummyLogger:
    """Swallows all SAC logging calls silently."""
    def log(self, *a, **kw): pass
    def log_histogram(self, *a, **kw): pass
    def log_param(self, *a, **kw): pass
    def dump(self, *a, **kw): pass


def build_agent_from_config(reward_type: str, config_path: str = "config",
                             config_name: str = "reacher") -> "SACAgent":
    """
    Reconstruct a SACAgent from the Hydra reacher config.
    This is exactly what run_reacher.py does at training time, guaranteeing
    the architecture dimensions are identical.
    """
    GlobalHydra.instance().clear()
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name,
                            overrides=[f"reward_type={reward_type}",
                                       f"device={DEVICE}"])

    # Probe a fresh env to get dimensions
    probe_env = ReacherWrapper(reward_type=reward_type, seed=0)
    obs_dim    = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    action_range = [
        float(probe_env.action_space.low.min()),
        float(probe_env.action_space.high.max()),
    ]
    probe_env.close()

    OmegaConf.set_struct(cfg, False)
    cfg.agent.obs_dim      = obs_dim
    cfg.agent.action_dim   = action_dim
    cfg.agent.action_range = action_range
    cfg.device             = DEVICE
    if 'critic_cfg' in cfg.agent:
        cfg.agent.critic_cfg.obs_dim    = obs_dim
        cfg.agent.critic_cfg.action_dim = action_dim
    if 'actor_cfg' in cfg.agent:
        cfg.agent.actor_cfg.obs_dim    = obs_dim
        cfg.agent.actor_cfg.action_dim = action_dim

    agent = hydra.utils.instantiate(cfg.agent)
    agent.train(False)   # put in eval mode immediately
    return agent


# ===========================================================================
# ▶  EVALUATION ENVIRONMENT SETUP
# ===========================================================================

def make_eval_env(reward_type: str) -> ReacherWrapper:
    """
    Create a ReacherWrapper with MAX_STEPS overriding the default 1000.

    For Ra / Rb: the wrapper's truncation boundary (step_count >= _MAX_EPISODE_STEPS)
    becomes 5000, so each episode runs for the full 5000 steps.

    For Rc:  the partial-reset trigger also shifts to every 5000 steps,
    meaning we get one big 5000-step window with no intermediate resets —
    exactly what we want for unbiased evaluation.
    """
    env = ReacherWrapper(reward_type=reward_type, seed=EVAL_ENV_SEED)
    # Patch the class-level constant on the *instance* so we don't affect
    # other instances that may share the class attribute.
    env._MAX_EPISODE_STEPS = MAX_STEPS
    return env


# ===========================================================================
# ▶  CORE EVALUATION LOOP
# ===========================================================================

def run_evaluation(agent, env: ReacherWrapper, policy_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run NUM_EPISODES episodes of MAX_STEPS steps each.

    Returns
    -------
    steps_to_goal   : np.ndarray, shape (NUM_EPISODES,)
        First timestep (1-indexed) the fingertip entered the target.
        Equals MAX_STEPS when the target was never reached.

    steps_in_target : np.ndarray, shape (NUM_EPISODES,)
        Total number of timesteps the fingertip spent inside the target.
    """
    print(f"\n{'─'*60}")
    print(f"  Evaluating policy: {policy_name}  |  {NUM_EPISODES} eps × {MAX_STEPS} steps")
    print(f"{'─'*60}")

    steps_to_goal_arr   = np.empty(NUM_EPISODES, dtype=np.int32)
    steps_in_target_arr = np.empty(NUM_EPISODES, dtype=np.int32)

    agent.train(False)   # deterministic eval mode

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()

        steps_to_goal   = MAX_STEPS   # sentinel: "never reached"
        steps_in_target = 0
        goal_reached    = False

        for step in range(1, MAX_STEPS + 1):
            # ── Deterministic action ─────────────────────────────────────────
            with torch.no_grad():
                action = agent.act(obs, sample=False)

            obs, reward, terminated, truncated, info = env.step(action)

            # ── In-target check ──────────────────────────────────────────────
            inside = is_in_target(env, info)

            if inside:
                # First crossing: record the timestep
                if not goal_reached:
                    steps_to_goal = step
                    goal_reached  = True
                # Every timestep inside counts
                steps_in_target += 1

            # ── Episode termination ──────────────────────────────────────────
            # For Ra/Rb: the env sets truncated=True at MAX_STEPS.
            # For Rc:    terminated=True only when the goal is reached with
            #            near-zero velocity; we never truncate in this script.
            if terminated or truncated:
                break

        steps_to_goal_arr[ep]   = steps_to_goal
        steps_in_target_arr[ep] = steps_in_target

        if (ep + 1) % 50 == 0 or ep == 0:
            reached_pct = 100.0 * np.mean(steps_to_goal_arr[:ep+1] < MAX_STEPS)
            mean_sit    = np.mean(steps_in_target_arr[:ep+1])
            print(
                f"  Ep {ep+1:>3d}/{NUM_EPISODES}  |  "
                f"reached={reached_pct:5.1f}%  |  "
                f"mean_steps_in_target={mean_sit:.1f}"
            )

    # ── Episode-level summary ────────────────────────────────────────────────
    reached_mask  = steps_to_goal_arr < MAX_STEPS
    n_reached     = reached_mask.sum()
    print(f"\n  ── Summary for {policy_name} ──")
    print(f"  Episodes reaching target : {n_reached}/{NUM_EPISODES}  ({100*n_reached/NUM_EPISODES:.1f}%)")
    if n_reached > 0:
        print(f"  Mean steps-to-goal (reached only) : {steps_to_goal_arr[reached_mask].mean():.1f}")
    print(f"  Mean steps-in-target (all eps)    : {steps_in_target_arr.mean():.1f}")

    agent.train(True)    # restore training mode
    return steps_to_goal_arr, steps_in_target_arr


# ===========================================================================
# ▶  MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  Q3 Reacher Evaluation  –  SAC policies (Ra, Rb, Rc)")
    print(f"  Episodes={NUM_EPISODES}  MaxSteps={MAX_STEPS}  Device={DEVICE}")
    print("=" * 60)

    # Map policy name → reward type character for env construction
    reward_type_map = {
        'reacher-a': 'a',
        'reacher-b': 'b',
        'reacher-c': 'c',
    }

    for policy_name, weights_path in POLICY_PATHS.items():
        reward_type = reward_type_map[policy_name]

        # ── Check weights file exists ────────────────────────────────────────
        if not os.path.exists(weights_path):
            print(f"\n⚠️  [{policy_name}] Weights file not found: {weights_path}")
            print(f"   Skipping.  (Save your trained agent with save_agent() and update POLICY_PATHS)")
            continue

        # ── Build agent ──────────────────────────────────────────────────────
        print(f"\n  [{policy_name}] Building agent from Hydra config …")
        agent = build_agent_from_config(reward_type)

        # ── Load weights ─────────────────────────────────────────────────────
        load_agent_weights(agent, weights_path)

        # ── Build evaluation environment (MAX_STEPS = 5000) ──────────────────
        env = make_eval_env(reward_type)

        # ── Run 500 evaluation episodes ──────────────────────────────────────
        stg, sit = run_evaluation(agent, env, policy_name)

        env.close()

        # ── Save metric arrays ───────────────────────────────────────────────
        stg_path = f"{policy_name}_steps_to_goal.npy"
        sit_path = f"{policy_name}_steps_in_target.npy"
        np.save(stg_path, stg)
        np.save(sit_path, sit)
        print(f"\n  ✅ Saved: {stg_path}  {stg.shape}")
        print(f"  ✅ Saved: {sit_path}  {sit.shape}")

    print("\n" + "=" * 60)
    print("  Evaluation complete.  Files ready for plotting.")
    print("=" * 60)


# ===========================================================================
# ▶  HOW TO ADD WEIGHT SAVING TO run_reacher.py
# ===========================================================================
# Insert the following two lines at the very end of the run() function in
# run_reacher.py, just before the results .npy is written:
#
#   from eval_q3_reacher import save_agent
#   save_agent(agent, f"reacher_{reward_type}_seed_{seed}_final.pt")
#
# Then update POLICY_PATHS in this file to point to the best seed's .pt file.
# ===========================================================================


if __name__ == '__main__':
    main()