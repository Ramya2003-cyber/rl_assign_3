import os
import numpy as np
import matplotlib.pyplot as plt

# ── NEW: Tell the script where to look ──
FOLDER = 'q3_pth'

# The base names of your files
AGENTS = ['reacher-a', 'reacher-b', 'reacher-c']
LABELS = [r'Teacher $\mathcal{R}_a$' + '\n(Dense Distance)', 
          r'Teacher $\mathcal{R}_b$' + '\n(Sparse Indicator)', 
          r'Teacher $\mathcal{R}_c$' + '\n(Time/Velocity)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

means_stg, ci_stg = [], []
means_sit, ci_sit = [], []

# Load data and calculate statistics
for agent in AGENTS:
    stg_path = os.path.join(FOLDER, f"{agent}_steps_to_goal.npy")
    sit_path = os.path.join(FOLDER, f"{agent}_steps_in_target.npy")
    
    try:
        stg = np.load(stg_path)
        sit = np.load(sit_path)
        
        # Means
        means_stg.append(np.mean(stg))
        means_sit.append(np.mean(sit))
        
        # 95% Confidence Interval: 1.96 * (std / sqrt(N))
        ci_stg.append(1.96 * (np.std(stg) / np.sqrt(len(stg))))
        ci_sit.append(1.96 * (np.std(sit) / np.sqrt(len(sit))))
    except FileNotFoundError:
        print(f"❌ Could not find {stg_path} or {sit_path}. Make sure the folder name is exactly correct!")
        exit()

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
x_pos = np.arange(len(AGENTS))

# --- Subplot 1: Steps to Goal (Lower is Better) ---
ax1.bar(x_pos, means_stg, yerr=ci_stg, capsize=8, color=COLORS, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(LABELS, fontsize=11)
ax1.set_title('Steps to Goal (Lower is Better)', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Timesteps', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
# Add a dashed line showing the 5000 max limit
ax1.axhline(5000, color='red', linestyle=':', alpha=0.5, label='Timeout (5000)')
ax1.legend()

# --- Subplot 2: Steps in Target (Higher is Better) ---
ax2.bar(x_pos, means_sit, yerr=ci_sit, capsize=8, color=COLORS, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(LABELS, fontsize=11)
ax2.set_title('Steps in Target (Higher is Better)', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Timesteps', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.set_ylim(0, 5200) # Max possible is 5000

plt.tight_layout()
plt.savefig('q3a_bar_charts.png', dpi=300)
print("✅ Success! Plot saved as q3a_bar_charts.png")