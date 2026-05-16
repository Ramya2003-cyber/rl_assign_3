#!/usr/bin/env python3


from __future__ import annotations

import os
import sys
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.reacher_custom import ReacherWrapper, _compute_distance
from core import utils


NUM_EPISODES: int   = 500
MAX_STEPS:    int   = 5000

POLICY_PATHS: dict[str, str] = {
    'reacher-a': 'reacher_a_seed_42_final.pt',
    'reacher-b': 'reacher_b_seed_42_final.pt',
    'reacher-c': 'reacher_c_seed_42_final.pt',

}

EVAL_ENV_SEED: int = 0
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

TARGET_RADIUS: float = 0.05  # must match ReacherWrapper._DIST_THRESHOLD


def is_in_target(env: ReacherWrapper, info: dict) -> bool:
    """Return True if the fingertip is currently inside the target sphere."""
    return bool(info.get('in_target', False))


def get_distance(env: ReacherWrapper) -> float:
    """Euclidean distance between fingertip and target (2-D task plane)."""
    return _compute_distance(env._physics)


def save_agent(agent, path: str):
    """Save SACAgent parameters to a .pt file."""
    torch.save({
        'actor_state_dict':         agent.actor.state_dict(),
        'critic_state_dict':        agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'log_alpha':                agent.log_alpha.detach().cpu(),
    }, path)
    print(f"  ✅ Agent weights saved → {path}")


def load_agent_weights(agent, path: str):
    """Load weights saved with save_agent() into an existing SACAgent."""
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


def build_agent_from_config(reward_type: str, config_path: str = "config",
                             config_name: str = "reacher") -> "SACAgent":
    """Reconstruct a SACAgent from the Hydra reacher config."""
    GlobalHydra.instance().clear()
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name,
                            overrides=[f"reward_type={reward_type}",
                                       f"device={DEVICE}"])

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
    agent.train(False)
    return agent


def make_eval_env(reward_type: str) -> ReacherWrapper:
    """Create eval ReacherWrapper with _MAX_EPISODE_STEPS patched to 5000."""
    env = ReacherWrapper(reward_type=reward_type, seed=EVAL_ENV_SEED)
    env._MAX_EPISODE_STEPS = MAX_STEPS
    return env


def run_evaluation(agent, env: ReacherWrapper, policy_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run NUM_EPISODES episodes; returns (steps_to_goal, steps_in_target) arrays."""
    print(f"\nEvaluating policy: {policy_name}")

    steps_to_goal_arr   = np.empty(NUM_EPISODES, dtype=np.int32)
    steps_in_target_arr = np.empty(NUM_EPISODES, dtype=np.int32)

    agent.train(False)

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()

        steps_to_goal   = MAX_STEPS
        steps_in_target = 0
        goal_reached    = False

        for step in range(1, MAX_STEPS + 1):
            with torch.no_grad():
                action = agent.act(obs, sample=False)

            obs, reward, terminated, truncated, info = env.step(action)

            inside = is_in_target(env, info)

            if inside:
                if not goal_reached:
                    steps_to_goal = step
                    goal_reached  = True
                steps_in_target += 1

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

    reached_mask  = steps_to_goal_arr < MAX_STEPS
    n_reached     = reached_mask.sum()
    print(f"\n  Summary for {policy_name}")
    print(f"  Episodes reaching target : {n_reached}/{NUM_EPISODES}  ({100*n_reached/NUM_EPISODES:.1f}%)")
    if n_reached > 0:
        print(f"  Mean steps-to-goal (reached only) : {steps_to_goal_arr[reached_mask].mean():.1f}")
    print(f"  Mean steps-in-target (all eps)    : {steps_in_target_arr.mean():.1f}")

    agent.train(True)
    return steps_to_goal_arr, steps_in_target_arr


def main():
    reward_type_map = {
        'reacher-a': 'a',
        'reacher-b': 'b',
        'reacher-c': 'c',
    }

    for policy_name, weights_path in POLICY_PATHS.items():
        reward_type = reward_type_map[policy_name]

        if not os.path.exists(weights_path):
            print(f"\n⚠️  [{policy_name}] Weights file not found: {weights_path}")
            continue

        print(f"\n  [{policy_name}] Building agent from Hydra config …")
        agent = build_agent_from_config(reward_type)

        load_agent_weights(agent, weights_path)

        env = make_eval_env(reward_type)

        stg, sit = run_evaluation(agent, env, policy_name)

        env.close()

        stg_path = f"{policy_name}_steps_to_goal.npy"
        sit_path = f"{policy_name}_steps_in_target.npy"
        np.save(stg_path, stg)
        np.save(sit_path, sit)
        print(f"\n  ✅ Saved: {stg_path}  {stg.shape}")
        print(f"  ✅ Saved: {sit_path}  {sit.shape}")

    print("\n" + "=" * 60)
    print("  Evaluation complete.  Files ready for plotting.")
    print("=" * 60)


if __name__ == '__main__':
    main()