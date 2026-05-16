import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = 'pebble-reacher' 
FILES = {
    r'Teacher $\mathcal{R}_a$ (Dense Distance)': 'pebble_reacher_a_seed_42.npy',
    r'Teacher $\mathcal{R}_b$ (Sparse Indicator)': 'pebble_reacher_b_seed_42.npy',
    r'Teacher $\mathcal{R}_c$ (Time/Velocity)': 'pebble_reacher_c_seed_42.npy'
}

plt.figure(figsize=(9, 6))

for label, filename in FILES.items():
    # Check folder first, fallback to current directory
    filepath = os.path.join(FOLDER, filename)
    if not os.path.exists(filepath) and os.path.exists(filename):
        filepath = filename

    if os.path.exists(filepath):
        # Load data and extract the dictionary
        data = np.load(filepath, allow_pickle=True).item()
        
        y = data['gt_returns']
        x = data['checkpoint_steps']
        
        # FIX: Average across the 3 evaluation episodes (axis=1), not the 51 checkpoints
        if y.ndim > 1: 
            if y.shape[0] == len(x):
                y = y.mean(axis=1) 
            else:
                y = y.mean(axis=0)
            
        plt.plot(x, y, label=label, linewidth=2.5)
        print(f"✅ Successfully plotted {filename}")
    else:
        print(f"⚠️ Could not find file: {filepath}")

plt.title('PEBBLE Reacher: Simulated Teacher Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Environment Steps', fontsize=12)
plt.ylabel('Ground Truth Return', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('q3_pebble_teachers_comparison.png', dpi=300)
print("\n🎉 DONE! Check your folder for q3_pebble_teachers_comparison.png")