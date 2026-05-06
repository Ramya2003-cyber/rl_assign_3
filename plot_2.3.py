#!/usr/bin/env python3
"""
plot_2.3.py
===========
Load all reacher .npy result files for a given reward_type and plot mean ± std
across seeds.

Usage:
    python plot_2.3.py --reward_type a
    python plot_2.3.py --reward_type b
    python plot_2.3.py --reward_type c
    python plot_2.3.py  # plots all three reward types in one figure

Result files are expected to follow the naming convention:
    reacher_results_<reward_type>_seed_<seed>.npy
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


REWARD_LABELS = {
    'a': r'$R_a$ (Distance Penalty)',
    'b': r'$R_b$ (Sparse)',
    'c': r'$R_c$ (Step Penalty)',
}
REWARD_COLORS = {'a': '#2196F3', 'b': '#4CAF50', 'c': '#F44336'}
EVAL_REWARD_NAMES = ['a', 'b', 'c']


def load_results(reward_type: str, results_dir: str = '.') -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load all seed files for ``reward_type`` and stack them.

    Returns
    -------
    (steps, stacked) where
      steps    : (C,) array of checkpoint steps
      stacked  : (num_seeds, C, 3, E) array of episode returns
    Returns None if no files found.
    """
    pattern = os.path.join(results_dir, f'reacher_results_{reward_type}_seed_*.npy')
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    all_returns = []
    steps = None
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        all_returns.append(data['returns'])          # (C, 3, E)
        if steps is None:
            steps = data['checkpoint_steps']         # (C,)

    stacked = np.stack(all_returns, axis=0)          # (S, C, 3, E)
    print(f"  Loaded {len(files)} seeds for reward_type={reward_type}  shape={stacked.shape}")
    return steps, stacked


def plot_reward_type(ax, steps, stacked, eval_reward_idx: int, training_reward_type: str):
    """
    Plot mean ± std across seeds for one (training, eval) reward pair.

    Parameters
    ----------
    ax : matplotlib axis
    steps : (C,) array
    stacked : (S, C, 3, E)
    eval_reward_idx : int   — index into axis-2 (0=Ra, 1=Rb, 2=Rc)
    training_reward_type : str  — 'a', 'b', or 'c'  (for colour/label)
    """
    # stacked[:, :, eval_reward_idx, :]  →  (S, C, E)
    data = stacked[:, :, eval_reward_idx, :]        # (S, C, E)
    # mean over episodes, then mean/std over seeds
    per_seed_mean = data.mean(axis=-1)              # (S, C)
    mean = per_seed_mean.mean(axis=0)               # (C,)
    std  = per_seed_mean.std(axis=0)                # (C,)

    color = REWARD_COLORS[training_reward_type]
    label = f"Train {REWARD_LABELS[training_reward_type]}"
    ax.plot(steps, mean, color=color, linewidth=2, label=label)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_type', type=str, default=None,
                        help="Which training reward type to plot ('a', 'b', 'c', or None for all).")
    parser.add_argument('--results_dir', type=str, default='.',
                        help="Directory containing the .npy result files.")
    args = parser.parse_args()

    reward_types_to_plot = ['a', 'b', 'c'] if args.reward_type is None else [args.reward_type]

    # ── One column per eval metric, one row per training type ────────────────
    fig, axes = plt.subplots(
        nrows=len(reward_types_to_plot),
        ncols=3,
        figsize=(18, 5 * len(reward_types_to_plot)),
        squeeze=False,
    )
    fig.suptitle('Reacher SAC — Cross-Evaluated Returns (mean ± std over seeds)', fontsize=14)

    for row_idx, train_rtype in enumerate(reward_types_to_plot):
        result = load_results(train_rtype, results_dir=args.results_dir)
        if result is None:
            print(f"  WARNING: no files found for reward_type={train_rtype}, skipping.")
            continue
        steps, stacked = result

        for col_idx, eval_rtype in enumerate(EVAL_REWARD_NAMES):
            ax = axes[row_idx][col_idx]
            plot_reward_type(ax, steps, stacked,
                             eval_reward_idx=col_idx,
                             training_reward_type=train_rtype)
            ax.set_title(f"Eval metric: {REWARD_LABELS[eval_rtype]}\n"
                         f"(trained on {REWARD_LABELS[train_rtype]})", fontsize=10)
            ax.set_xlabel('Training steps')
            ax.set_ylabel('Mean episode return')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = 'reacher_cross_eval.png'
    plt.savefig(out_fig, dpi=150)
    print(f"\n  Figure saved → {out_fig}")
    plt.show()


if __name__ == '__main__':
    main()
