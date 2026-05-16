import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 7)

def load_and_smooth_train(base_path, label, window=50):
    """Loads all train.csv files, smooths them, and bins steps for alignment."""
    path_pattern = os.path.join(base_path, 'exp_local/LunarLander-v3/*/train.csv')
    files = glob.glob(path_pattern)
    
    print(f"Loading Training data for {label}: Found {len(files)} files.")
    
    all_dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df = df[['step', 'episode_reward']].dropna()
        
        df['smoothed_reward'] = df['episode_reward'].rolling(window=window, min_periods=1).mean()

        df['step_bin'] = (df['step'] // 5000) * 5000
        
        df['seed'] = i
        df['Algorithm'] = label
        all_dfs.append(df)
    
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

def generate_training_plot():
    sac_train = load_and_smooth_train('discrete_sac', 'Discrete SAC')
    dqn_train = load_and_smooth_train('dqn', 'DQN')

    if sac_train.empty or dqn_train.empty:
        print("Error: could not find training data. Check folder paths.")
        return

    combined_data = pd.concat([sac_train, dqn_train])

    plt.figure()
    sns.lineplot(data=combined_data, x='step_bin', y='smoothed_reward', 
                 hue='Algorithm', errorbar='ci', linewidth=2)

    plt.axhline(y=200, color='red', linestyle='--', alpha=0.4, label='Solved Threshold')

    plt.title('Question 2.2.4: Training Performance Comparison (15 Seeds)', fontweight='bold')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Smoothed Episode Return (50-ep window)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    output_path = '../../images/q2_2_4_training_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_training_plot()
