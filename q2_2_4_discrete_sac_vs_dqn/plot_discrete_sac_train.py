import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid")

def plot_sac_train_with_ci():
    path = 'discrete_sac/exp_local/LunarLander-v3/discrete_sac_seed_*/train.csv'
    files = glob.glob(path)
    print(f"Found {len(files)} seeds for Discrete SAC Training.")

    all_dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']].dropna()
        df['step_bin'] = (df['step'] // 10000) * 10000
        df['seed'] = i
        all_dfs.append(df)

    if not all_dfs:
        print("No data found!")
        return

    data = pd.concat(all_dfs)

    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=data, x='step_bin', y='episode_reward', errorbar='ci', label='Discrete SAC (Train)')

    plt.title('Discrete SAC Training Performance (15 Seeds)')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Average Return per Episode')
    plt.axhline(y=200, color='red', linestyle='--', label='Solved Threshold')
    plt.legend()
    output_path = '../../images/q2_2_4_discrete_sac_training.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_sac_train_with_ci()
