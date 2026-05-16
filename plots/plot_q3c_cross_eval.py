import os
import numpy as np
import matplotlib.pyplot as plt

# Folders where your 15 seeds are stored
AGENT_FOLDERS = {
    r'SAC-$\mathcal{R}_a$ (Dense Agent)': 'reacher-a',
    r'SAC-$\mathcal{R}_b$ (Sparse Agent)': 'reacher-b',
    r'SAC-$\mathcal{R}_c$ (Time Agent)': 'reacher-c'
}

# The short keys your script used to save the files
REWARD_TYPES = ['a', 'b', 'c']
NUM_SEEDS = 15

# Evaluation metric labels for the 3 separate plots
METRIC_LABELS = {
    0: r'Evaluated under $\mathcal{R}_a$ (Dense Distance Metric)',
    1: r'Evaluated under $\mathcal{R}_b$ (Sparse Indicator Metric)',
    2: r'Evaluated under $\mathcal{R}_c$ (Time/Velocity Metric)'
}

def load_agent_data(folder, reward_type):
    """Loads and averages all 15 seeds for a given agent folder."""
    all_seeds_returns = []
    checkpoint_steps = None
    
    for seed in range(NUM_SEEDS):
        filename = f"reacher_results_{reward_type}_seed_{seed}.npy"
        filepath = os.path.join(folder, filename)
        
        # Fallback if files are sitting in the root directory instead of folders
        if not os.path.exists(filepath):
            filepath = f"reacher_results_{reward_type}_seed_{seed}.npy"
            
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True).item()
            # shape of returns: (Checkpoints, 3, Episodes)
            returns = data['returns'] 
            checkpoint_steps = data['checkpoint_steps']
            
            # Average over the evaluation episodes (axis=2) -> shape becomes (Checkpoints, 3)
            mean_episodes = returns.mean(axis=2)
            all_seeds_returns.append(mean_episodes)
            
    if len(all_seeds_returns) == 0:
        return None, None
        
    # Average over all 15 seeds (axis=0) -> shape becomes (Checkpoints, 3)
    final_data = np.mean(all_seeds_returns, axis=0)
    return checkpoint_steps, final_data

# Load data for all 3 agents
agent_data = {}
steps = None

for agent_label, folder in AGENT_FOLDERS.items():
    r_type = folder.split('-')[-1] # extracts 'a', 'b', or 'c'
    chk_steps, data = load_agent_data(folder, r_type)
    if data is not None:
        agent_data[agent_label] = data
        steps = chk_steps

if steps is None:
    raise FileNotFoundError("Could not find your training .npy files. Double check your file paths!")

# Generate the 3 separate figures required by the TA
for metric_idx in range(3):
    plt.figure(figsize=(8, 5))
    
    for agent_label, data in agent_data.items():
        # data has shape (Checkpoints, 3). Extract the column for the current evaluation metric.
        y_values = data[:, metric_idx]
        plt.plot(steps, y_values, label=agent_label, linewidth=2.5)
        
    plt.title(f'Cross Evaluation: {METRIC_LABELS[metric_idx]}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Environment Steps', fontsize=10)
    plt.ylabel('Average Evaluation Return', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    filename = f'reacher_cross_eval_metric_{REWARD_TYPES[metric_idx]}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"💾 Saved separate figure: {filename}")

print("\n✅ All 3 cross-evaluation plots generated successfully!")