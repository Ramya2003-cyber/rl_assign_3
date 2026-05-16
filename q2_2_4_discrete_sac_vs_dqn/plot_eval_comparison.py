import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (12, 7)

def load_and_align_data(base_path, label):
    """Loads eval.csv and aligns steps into 10k bins for clean CI shading."""
    path_pattern = os.path.join(base_path, 'exp_local/LunarLander-v3/*/eval.csv')
    files = glob.glob(path_pattern)
    
    print(f"Loading {label}: Found {len(files)} seeds.")
    
    all_dfs = []
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={'reward': 'episode_reward', 'value': 'episode_reward'})
            df = df[['step', 'episode_reward']].dropna()

            df['step'] = (df['step'] // 10000) * 10000

            df['seed'] = i
            df['Algorithm'] = label
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

def generate_eval_comparison_plot():
    sac_data = load_and_align_data('discrete_sac', 'Discrete SAC')
    dqn_data = load_and_align_data('dqn', 'DQN')

    if sac_data.empty or dqn_data.empty:
        print("Error: missing evaluation data. Check the discrete SAC and DQN folders.")
        return

    combined = pd.concat([sac_data, dqn_data])

    plt.figure()
    sns.lineplot(data=combined, x='step', y='episode_reward', hue='Algorithm', 
                 errorbar='ci', linewidth=2.5)

    plt.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Solved Threshold (+200)')

    plt.title('Question 2.2.4: Discrete SAC vs. DQN Comparison (15 Seeds Each)', fontweight='bold')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Average Undiscounted Return (20 Eval Episodes)')
    plt.legend(loc='lower right')
    plt.xlim(0, 300000)
    plt.tight_layout()

    output_path = '../../images/q2_2_4_discrete_sac_vs_dqn_eval.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_eval_comparison_plot()
