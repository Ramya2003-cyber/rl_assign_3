import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration matching your 51 evaluation points (0 to 500k steps)
STEPS = np.linspace(0, 500000, 51) 

# Mapping of folder to its corresponding evaluation index in the second dimension of the array
# Index 0 = R_a, Index 1 = R_b, Index 2 = R_c
CONFIGS = {
    'reacher-a': {
        'label': r'SAC-$\mathcal{R}_a$ evaluated on $\mathcal{R}_a$',
        'idx': 0,
        'filename': 'reacher_ra_performance.png',
        'title': r'Reacher Performance: SAC-$\mathcal{R}_a$ (Dense Distance Penalty)',
        'color': '#1f77b4'
    },
    'reacher-b': {
        'label': r'SAC-$\mathcal{R}_b$ evaluated on $\mathcal{R}_b$',
        'idx': 1,
        'filename': 'reacher_rb_performance.png',
        'title': r'Reacher Performance: SAC-$\mathcal{R}_b$ (Sparse Indicator Reward)',
        'color': '#ff7f0e'
    },
    'reacher-c': {
        'label': r'SAC-$\mathcal{R}_c$ evaluated on $\mathcal{R}_c$',
        'idx': 2,
        'filename': 'reacher_rc_performance.png',
        'title': r'Reacher Performance: SAC-$\mathcal{R}_c$ (Time-to-Target Penalty)',
        'color': '#2ca02c'
    }
}

def load_clean_data(folder_path, eval_idx):
    aggregated_runs = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder {folder_path} not found.")
        return None

    for file in os.listdir(folder_path):
        if file.endswith('.npy') and 'results' in file:
            try:
                filepath = os.path.join(folder_path, file)
                raw_data = np.load(filepath, allow_pickle=True).item()
                returns = raw_data['returns']  # Shape: (51, 3, 20)
                
                # 1. Average over the 20 evaluation episodes (axis 2) -> Shape: (51, 3)
                mean_over_episodes = np.mean(returns, axis=2)
                
                # 2. Extract only the self-evaluation stream (axis 1) -> Shape: (51,)
                self_eval_stream = mean_over_episodes[:, eval_idx]
                aggregated_runs.append(self_eval_stream)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    return np.array(aggregated_runs) if len(aggregated_runs) > 0 else None

def generate_separate_plots():
    for folder, setup in CONFIGS.items():
        data_matrix = load_clean_data(folder, setup['idx'])
        
        if data_matrix is None:
            print(f"❌ Skipping {folder} due to missing data arrays.")
            continue
            
        print(f"📈 Processing {data_matrix.shape[0]} seeds for {folder} folder...")
        
        # Calculate mean and standard deviation across seeds
        mean_curve = np.mean(data_matrix, axis=0)
        std_curve = np.std(data_matrix, axis=0)
        
        # Initialize a brand new standalone figure
        plt.figure(figsize=(7, 5))
        
        # Plot mean curve and shaded variance band
        plt.plot(STEPS, mean_curve, label=setup['label'], color=setup['color'], lw=2.5)
        plt.fill_between(STEPS, mean_curve - std_curve, mean_curve + std_curve, 
                         alpha=0.18, color=setup['color'])
        
        # Labeling and layout adjustments
        plt.title(setup['title'], fontsize=11, fontweight='bold', pad=12)
        plt.xlabel('Environment Steps', fontsize=10)
        plt.ylabel('Average Return', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.xlim(-10000, 510000) # Lock x-axis limits exactly the same for all three
        plt.tight_layout()
        
        # Save figure file
        plt.savefig(setup['filename'], dpi=300)
        plt.close()
        print(f"💾 Saved: {setup['filename']}")

if __name__ == '__main__':
    generate_separate_plots()
    print("\n🏁 All three figures generated successfully!")