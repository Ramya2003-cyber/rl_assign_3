import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = 'pebble-pendulum'
ANGLES = ['-60', '90', '120']
BUDGETS = [100, 500, 1000]
NUM_SEEDS = 15

for angle in ANGLES:
    # Create a fresh, separate figure for each angle
    plt.figure(figsize=(8, 5)) 
    
    for budget in BUDGETS:
        folder_name = f'pebble_{angle}_budget_{budget}'
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        all_seeds_data = []
        x_steps = None # To store the actual environment steps
        
        for seed in range(NUM_SEEDS):
            filename = f'pebble_results_angle_{angle}_budget_{budget}_seed_{seed}.npy'
            filepath = os.path.join(folder_path, filename)
            
            # Fallback to current directory if folder doesn't exist
            if not os.path.exists(filepath) and os.path.exists(filename):
                filepath = filename
                
            if os.path.exists(filepath):
                data = np.load(filepath, allow_pickle=True)
                
                # Handle both dict-based saves and raw arrays
                if data.ndim == 0 and isinstance(data.item(), dict):
                    d = data.item()
                    y = d.get('gt_returns', d.get('returns'))
                    
                    # Safely extract the real steps if they exist!
                    if 'checkpoint_steps' in d:
                        x_steps = d['checkpoint_steps']
                    elif 'steps' in d:
                        x_steps = d['steps']
                else:
                    y = data
                
                # Squeeze/mean if it saved multiple eval episodes per checkpoint
                if y.ndim > 1:
                    if y.shape[0] > y.shape[1]:
                        y = y.mean(axis=1) 
                    else:
                        y = y.mean(axis=0)
                all_seeds_data.append(y)
                
        if all_seeds_data:
            mean_y = np.mean(all_seeds_data, axis=0)
            
            # THE FIX: Use actual steps, or correctly map to 50,000
            if x_steps is not None and len(x_steps) == len(mean_y):
                x = x_steps
            else:
                x = np.linspace(0, 50000, len(mean_y)) 
                
            plt.plot(x, mean_y, label=f'Budget: {budget}', linewidth=2.5)

    # Styling the individual plot
    plt.title(f'PEBBLE Budget Comparison (Target: {angle}°)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Ground Truth Return', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save each plot separately
    save_name = f'q2_pebble_budget_{angle}deg.png'
    plt.savefig(save_name, dpi=300)
    plt.close() 
    print(f"✅ Saved corrected 50k plot: {save_name}")

print("\n🎉 All individual budget plots generated successfully!")