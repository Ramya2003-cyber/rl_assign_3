import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 7)

def load_and_smooth_hover(path_pattern, label, window=50):
    """Load train.csv, smooth noisy episode returns, and align seeds on 10k bins."""
    files = glob.glob(path_pattern)
    print(f"Loading {label}: Found {len(files)} seeds.")

    all_dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']].dropna()
        df['smoothed_reward'] = df['episode_reward'].rolling(window=window, min_periods=1).mean()

        df['step_bin'] = (df['step'] // 10000) * 10000

        df['seed'] = i
        df['Variant'] = label
        all_dfs.append(df)

    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

def generate_hover_plot():
    auto_pattern = 'exp_local/LunarLanderContinuous-v3/hover_auto_seed_*/train.csv'
    fixed_pattern = 'exp_local/LunarLanderContinuous-v3/hover_fixed_seed_*/train.csv'

    auto_data = load_and_smooth_hover(auto_pattern, 'Automated alpha')
    fixed_data = load_and_smooth_hover(fixed_pattern, 'Fixed alpha = 0.01')

    if auto_data.empty or fixed_data.empty:
        print("Error: missing hover training data. Check that both variants have all seeds.")
        return

    combined = pd.concat([auto_data, fixed_data])

    plt.figure()
    sns.lineplot(
        data=combined,
        x='step_bin',
        y='smoothed_reward',
        hue='Variant',
        errorbar='ci',
        linewidth=2.5,
    )

    plt.axvline(x=300000, color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(310000, -250, 'Reward flip\n+200 to -100', color='black', fontweight='bold', fontsize=12)

    plt.title('Question 2.2.3: SAC Robustness to Reward Change (15 Seeds)', fontweight='bold')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Smoothed Episode Return (50-episode window)')
    plt.legend(loc='lower left')
    plt.tight_layout()

    output_path = '../../../images/q2_2_3_hover_reward_flip.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_hover_plot()
