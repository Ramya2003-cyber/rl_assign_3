import numpy as np
import matplotlib.pyplot as plt
import os

def plot_q2_results():
    # The 8 angles we ran
    angles = [0, -10, 30, -60, 90, -90, 120, -150]
    
    # Pendulum-v1 default evaluation: 100k steps, eval every 10k
    # This matches the 'seed_evals' length of 10 in your run_pendulum.py
    steps = np.arange(10000, 110000, 10000)
    
    plt.figure(figsize=(12, 8))
    
    for angle in angles:
        filename = f"results_angle_{angle}_auto_True_alpha_0.1_scale_1.0.npy"
        
        if os.path.exists(filename):
            # Load data: shape is (num_seeds, 10)
            data = np.load(filename)
            
            # Calculate mean and standard deviation across seeds (axis 0)
            mean_return = np.mean(data, axis=0)
            std_return = np.std(data, axis=0)
            
            # Plot the mean line
            line, = plt.plot(steps, mean_return, label=f'Target: {angle}°')
            
            # Add the shaded error region (Standard Deviation)
            plt.fill_between(steps, 
                             mean_return - std_return, 
                             mean_return + std_return, 
                             color=line.get_color(), 
                             alpha=0.2)
        else:
            print(f"Warning: {filename} not found. Skipping...")

    plt.title("SAC Performance across Different Target Angles", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Return (20 Eval Episodes)", fontsize=12)
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot for your report
    plt.savefig("q2_target_angles_comparison.png", dpi=300)
    print("Plot saved as q2_target_angles_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_q2_results()