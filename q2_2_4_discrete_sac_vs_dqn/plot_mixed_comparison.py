import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 7)

def load_and_prepare_data():
    all_data = []

    dqn_files = glob.glob('dqn/exp_local/LunarLander-v3/dqn_seed_*/eval.csv')
    print(f"Found {len(dqn_files)} seeds for DQN (using eval.csv)")
    for i, f in enumerate(dqn_files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']]
        df['step_bin'] = (df['step'] // 10000) * 10000
        df['Variant'] = 'DQN'
        df['seed'] = i
        all_data.append(df)

    sac_files = glob.glob('discrete_sac/exp_local/LunarLander-v3/discrete_sac_seed_*/train.csv')
    print(f"Found {len(sac_files)} seeds for Discrete SAC (using train.csv + smoothing)")
    for i, f in enumerate(sac_files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']].dropna()
        df['episode_reward'] = df['episode_reward'].rolling(window=50, min_periods=1).mean()
        df['step_bin'] = (df['step'] // 10000) * 10000
        df['Variant'] = 'Discrete SAC'
        df['seed'] = i
        all_data.append(df)

    return pd.concat(all_data) if all_data else pd.DataFrame()

def plot_comparison():
    data = load_and_prepare_data()
    if data.empty:
        print("No data found. Check your folder paths.")
        return

    plt.figure()
    sns.lineplot(data=data, x='step_bin', y='episode_reward', hue='Variant', errorbar='ci', linewidth=2.5)

    plt.title('Question 2.2.4: Discrete SAC vs. DQN Comparison', fontweight='bold')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Average Undiscounted Return')
    plt.axhline(y=200, color='black', linestyle='--', alpha=0.3, label='Solved Threshold')
    plt.legend()

    output_path = '../../images/q2_2_4_final_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_comparison()
