import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 6)

def load_and_smooth(path_pattern, label, window=50):
    """Load train.csv, smooth noisy episode returns, and align seeds by timestep."""
    files = glob.glob(path_pattern)
    print(f"Loading {len(files)} seeds for {label}...")

    all_data = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']].dropna()
        df['smoothed_reward'] = df['episode_reward'].rolling(window=window, min_periods=1).mean()

        df['step_bin'] = (df['step'] // 5000) * 5000

        df['seed'] = i
        df['Variant'] = label
        all_data.append(df)

    return pd.concat(all_data) if all_data else pd.DataFrame()

def make_final_baseline_plot():
    data = load_and_smooth('exp_local/LunarLanderContinuous-v3/baseline_seed_*/train.csv', 'SAC baseline')

    if data.empty:
        print("No data found. Check exp_local/LunarLanderContinuous-v3/baseline_seed_*/train.csv")
        return

    plt.figure()
    sns.lineplot(data=data, x='step_bin', y='smoothed_reward', errorbar='ci', linewidth=2.5)

    plt.title('Question 2.2.1: Continuous SAC Baseline (15 Seeds)', fontweight='bold')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Smoothed Episode Return (50-episode window)')
    plt.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Solved Threshold')
    plt.legend()

    output_path = '../../../images/q2_2_1_continuous_sac_baseline.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    make_final_baseline_plot()
