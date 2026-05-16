#!/usr/bin/env python3


from __future__ import annotations

import os
import sys
import time
import random
from typing import Literal

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# ── Make sure the project root is importable ────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.reacher_custom import ReacherWrapper
from agent.reward_model import RewardModel, SimulatedTeacher, PreferenceDataset, PEBBLEReplayBuffer
from core import utils



class DummyLogger:
    def log(self, *a, **kw): pass
    def log_histogram(self, *a, **kw): pass
    def log_param(self, *a, **kw): pass
    def dump(self, *a, **kw): pass



def evaluate_gt_returns(
    agent,
    reward_type: str,
    num_episodes: int = 10,
    rc_step_limit: int = 1_000,
    rc_timeout_penalty: float = -20.0,
) -> dict[str, float]:
   
    eval_env = ReacherWrapper(reward_type=reward_type, seed=0)
    sums = {"a": [], "b": [], "c": []}

    agent.train(False)

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        ep = {"a": 0.0, "b": 0.0, "c": 0.0}
        done = False
        step_in_ep = 0
        rc_truncated = False

        while not done:
            with torch.no_grad():
                action = agent.act(obs, sample=False)
            obs, _, terminated, truncated, info = eval_env.step(action)
            all_r = info.get("all_rewards", {})
            for k in ("a", "b", "c"):
                ep[k] += all_r.get(k, 0.0)

            step_in_ep += 1
            done = terminated or truncated
            if step_in_ep >= rc_step_limit and not done:
                rc_truncated = True
                done = True

        if rc_truncated:
            ep["c"] += rc_timeout_penalty

        for k in ("a", "b", "c"):
            sums[k].append(ep[k])

    eval_env.close()
    agent.train(True)
    return {k: float(np.mean(v)) for k, v in sums.items()}



def pebble_sac_update(agent, replay_buffer: PEBBLEReplayBuffer, logger, step: int):
    """One SAC gradient step using the (relabelled) buffer rewards."""
    obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(agent.batch_size)

    agent.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

    if step % agent.actor_update_frequency == 0:
        agent.update_actor_and_alpha(obs, logger, step)

    if step % agent.critic_target_update_frequency == 0:
        utils.soft_update_params(agent.critic, agent.critic_target, agent.critic_tau)



def update_reward_model(
    reward_model: RewardModel,
    teacher: SimulatedTeacher,
    pref_dataset: PreferenceDataset,
    replay_buffer: PEBBLEReplayBuffer,
    seg_len: int,
    queries_per_update: int,
    rm_epochs: int,
    rm_batch_size: int,
    verbose: bool = True,
) -> dict | None:
   
    # ── Gather new preferences ────────────────────────────────────────────────
    new_labels = 0
    for _ in range(queries_per_update):
        if teacher.budget_exhausted:
            break
        n = replay_buffer.capacity if replay_buffer.full else replay_buffer.idx
        if n < seg_len * 2:
            break

        seg1 = replay_buffer.sample_segment(seg_len)
        seg2 = replay_buffer.sample_segment(seg_len)

        label = teacher.query(seg1["gt_rewards"], seg2["gt_rewards"])
        if label is None:
            break

        pref_dataset.add(
            seg1["states"], seg1["actions"],
            seg2["states"], seg2["actions"],
            label,
        )
        new_labels += 1

    if len(pref_dataset) < 2:
        return None

    # ── Train RM ────────────────────────────────────────────────────────────
    history = reward_model.fit(
        pref_dataset,
        num_epochs=rm_epochs,
        batch_size=rm_batch_size,
    )

    # ── Relabel the replay buffer ────────────────────────────────────────────
    reward_model.relabel_buffer(replay_buffer)

    if verbose:
        train_acc = history["train_acc"][-1] if history["train_acc"] else float("nan")
        val_acc   = history["val_acc"][-1]   if history["val_acc"]   else float("nan")
        train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
        val_loss   = history["val_loss"][-1]   if history["val_loss"]   else float("nan")
        print(
            f"  [RM] new_labels={new_labels:3d}  |  "
            f"dataset_size={len(pref_dataset):5d}  |  "
            f"budget_left={teacher.budget_remaining:5d}  |  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )

    return history



def run(cfg: DictConfig):
    reward_type: str = cfg.reward_type
    seed:        int = int(cfg.seed)
    device_str:  str = cfg.get("device", "cpu")

    num_train_steps:      int = int(cfg.get("num_train_steps",      500_000))
    num_seed_steps:       int = int(cfg.get("num_seed_steps",        10_000))
    eval_frequency:       int = int(cfg.get("eval_frequency",        10_000))
    num_eval_episodes:    int = int(cfg.get("num_eval_episodes",         10))
    replay_buffer_cap:    int = int(cfg.get("replay_buffer_capacity", 1_000_000))

    pebble_cfg                = cfg.get("pebble", {})
    max_feedback:       int   = int(pebble_cfg.get("max_feedback",       2_000))
    seg_len:            int   = int(pebble_cfg.get("seg_len",               50))
    feedback_freq:      int   = int(pebble_cfg.get("feedback_freq",       5_000))  # steps between RM updates
    queries_per_update: int   = int(pebble_cfg.get("queries_per_update",     20))
    rm_pretrain_queries:int   = int(pebble_cfg.get("rm_pretrain_queries",   100))
    rm_epochs:          int   = int(pebble_cfg.get("rm_epochs",              50))
    rm_pretrain_epochs: int   = int(pebble_cfg.get("rm_pretrain_epochs",    200))
    rm_batch_size:      int   = int(pebble_cfg.get("rm_batch_size",          64))
    rm_hidden_dim:      int   = int(pebble_cfg.get("rm_hidden_dim",         256))
    rm_lr:              float = float(pebble_cfg.get("rm_lr",              3e-4))
    pref_buffer_cap:    int   = int(pebble_cfg.get("pref_buffer_capacity", 5_000))

    print(f"\n{'='*65}")
    print(f"  PEBBLE-Reacher  |  reward_type={reward_type}  |  seed={seed}")
    print(f"  device={device_str}  |  total_steps={num_train_steps}")
    print(f"  max_feedback={max_feedback}  seg_len={seg_len}  feedback_freq={feedback_freq}")
    print(f"{'='*65}\n")

    # ── Seeding ──────────────────────────────────────────────────────────────
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

    # ── SAC agent ─────────────────────────────────────────────────────────────
    OmegaConf.set_struct(cfg, False)
    cfg.agent.obs_dim      = obs_dim
    cfg.agent.action_dim   = action_dim
    cfg.agent.action_range = action_range
    if "critic_cfg" in cfg.agent:
        cfg.agent.critic_cfg.obs_dim    = obs_dim
        cfg.agent.critic_cfg.action_dim = action_dim
    if "actor_cfg" in cfg.agent:
        cfg.agent.actor_cfg.obs_dim    = obs_dim
        cfg.agent.actor_cfg.action_dim = action_dim
    agent = hydra.utils.instantiate(cfg.agent)

    # ── Replay buffer (PEBBLE extended) ──────────────────────────────────────
    replay_buffer = PEBBLEReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=replay_buffer_cap,
        device=device_str,
    )

    # ── PEBBLE components ─────────────────────────────────────────────────────
    reward_model = RewardModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=rm_hidden_dim,
        lr=rm_lr,
        device=device_str,
        val_fraction=0.1,
    )
    teacher = SimulatedTeacher(
        reward_type=reward_type,
        max_feedback=max_feedback,
    )
    pref_dataset = PreferenceDataset(capacity=pref_buffer_cap)

    logger = DummyLogger()

    gt_returns_log   = []      # list of dicts {'a': float, 'b': float, 'c': float}
    checkpoint_steps = []
    rm_history_log   = []      # one entry per RM update; each is fit() history dict

    print("  [Eval @ step 0]")
    gt = evaluate_gt_returns(agent, reward_type, num_eval_episodes)
    gt_returns_log.append(gt)
    checkpoint_steps.append(0)
    print(f"  GT returns  Ra={gt['a']:.2f}  Rb={gt['b']:.2f}  Rc={gt['c']:.2f}")

    global_step        = 0
    next_eval_at       = eval_frequency
    next_feedback_at   = num_seed_steps + feedback_freq   # first RM update after seed phase
    rm_pretrained      = False

    obs, _ = env.reset()
    ep_reward  = 0.0
    ep_step    = 0
    ep_count   = 0

    while global_step < num_train_steps:

        # ── 1. Action selection ──────────────────────────────────────────────
        if global_step < num_seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # ── 2. Environment step ──────────────────────────────────────────────
        next_obs, gt_reward, terminated, truncated, info = env.step(action)
        episode_ended  = terminated or truncated
        done_for_buf   = float(terminated)
        done_no_max    = float(terminated)

        # Store transition; rewards slot starts with GT reward (will be relabelled later)
        replay_buffer.add(obs, action, gt_reward, next_obs, done_for_buf, done_no_max)

        ep_reward   += gt_reward
        ep_step     += 1
        global_step += 1

        # ── 3. Phase 2: Pre-train reward model right after seed phase ─────────
        if global_step == num_seed_steps and not rm_pretrained:
            print(f"\n{'─'*65}")
            print(f"  [PEBBLE Phase 2] Pre-training Reward Model with {rm_pretrain_queries} queries …")
            n_buf = replay_buffer.capacity if replay_buffer.full else replay_buffer.idx
            if n_buf >= seg_len * 2:
                # Collect initial preference labels
                for _ in range(rm_pretrain_queries):
                    if teacher.budget_exhausted:
                        break
                    seg1 = replay_buffer.sample_segment(seg_len)
                    seg2 = replay_buffer.sample_segment(seg_len)
                    label = teacher.query(seg1["gt_rewards"], seg2["gt_rewards"])
                    if label is None:
                        break
                    pref_dataset.add(
                        seg1["states"], seg1["actions"],
                        seg2["states"], seg2["actions"],
                        label,
                    )

                if len(pref_dataset) >= 2:
                    history = reward_model.fit(
                        pref_dataset,
                        num_epochs=rm_pretrain_epochs,
                        batch_size=rm_batch_size,
                    )
                    reward_model.relabel_buffer(replay_buffer)
                    rm_history_log.append(history)
                    tr_acc = history["train_acc"][-1] if history["train_acc"] else float("nan")
                    vl_acc = history["val_acc"][-1]   if history["val_acc"]   else float("nan")
                    print(
                        f"  Pre-training done.  dataset={len(pref_dataset)}  "
                        f"budget_left={teacher.budget_remaining}  "
                        f"train_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}"
                    )
            else:
                print(f"  Skipping pre-training: only {n_buf} transitions in buffer.")

            rm_pretrained = True
            print(f"{'─'*65}\n")

        # ── 4. Phase 3: SAC updates (after seed phase) ───────────────────────
        if global_step >= num_seed_steps and len(replay_buffer) >= agent.batch_size:
            pebble_sac_update(agent, replay_buffer, logger, global_step)

        # ── 5. Periodic RM update + buffer relabelling ────────────────────────
        if global_step >= next_feedback_at and not teacher.budget_exhausted:
            print(f"\n  [RM Update @ step {global_step}]")
            hist = update_reward_model(
                reward_model=reward_model,
                teacher=teacher,
                pref_dataset=pref_dataset,
                replay_buffer=replay_buffer,
                seg_len=seg_len,
                queries_per_update=queries_per_update,
                rm_epochs=rm_epochs,
                rm_batch_size=rm_batch_size,
                verbose=True,
            )
            if hist is not None:
                rm_history_log.append(hist)
            next_feedback_at += feedback_freq

        # ── 6. Periodic evaluation ────────────────────────────────────────────
        if global_step >= next_eval_at:
            print(f"\n  [Eval @ step {global_step}]  (ep={ep_count})")
            gt = evaluate_gt_returns(agent, reward_type, num_eval_episodes)
            gt_returns_log.append(gt)
            checkpoint_steps.append(global_step)
            print(
                f"  GT returns  Ra={gt['a']:.3f}  Rb={gt['b']:.3f}  Rc={gt['c']:.3f}  "
                f"budget_left={teacher.budget_remaining}"
            )
            next_eval_at += eval_frequency

        # ── 7. Episode bookkeeping ────────────────────────────────────────────
        if episode_ended:
            if ep_count % 50 == 0 or global_step < num_seed_steps + 1000:
                print(
                    f"  Ep {ep_count:5d}  |  step {global_step:7d}  |  "
                    f"ep_steps {ep_step:5d}  |  ep_gt_reward {ep_reward:8.2f}"
                )
            ep_count  += 1
            ep_reward  = 0.0
            ep_step    = 0
            obs, _     = env.reset()
        else:
            obs = next_obs

    # ── Final evaluation ──────────────────────────────────────────────────────
    if checkpoint_steps[-1] < global_step:
        print(f"\n  [Final Eval @ step {global_step}]")
        gt = evaluate_gt_returns(agent, reward_type, num_eval_episodes)
        gt_returns_log.append(gt)
        checkpoint_steps.append(global_step)
        print(f"  GT returns  Ra={gt['a']:.3f}  Rb={gt['b']:.3f}  Rc={gt['c']:.3f}")

    env.close()

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = f"pebble_reacher_{reward_type}_seed_{seed}.npy"

    # Build condensed accuracy arrays from rm_history_log
    rm_train_accs = [h["train_acc"] for h in rm_history_log]
    rm_val_accs   = [h["val_acc"]   for h in rm_history_log]

    # gt_returns: (C, 3)  – checkpoints × reward-type columns (a, b, c)
    gt_arr = np.array([[g["a"], g["b"], g["c"]] for g in gt_returns_log])

    np.save(out_path, {
        "gt_returns":        gt_arr,
        "checkpoint_steps":  np.array(checkpoint_steps),
        "rm_train_accs":     rm_train_accs,
        "rm_val_accs":       rm_val_accs,
        "reward_type":       reward_type,
        "seed":              seed,
        "max_feedback":      max_feedback,
    })
    print(f"\n  Results saved → {out_path}")
    print(f"  gt_returns shape: {gt_arr.shape}  (checkpoints × reward_types)")
    print("  Done.\n")


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="pebble_reacher", version_base=None)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
