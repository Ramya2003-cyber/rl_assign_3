import os
import numpy as np
import matplotlib.pyplot as plt

ANGLES = [-150, -60, 90, 120]
ALPHAS = ['0.01', '0.05', '0.1', '0.2', '0.5']
STEPS = np.linspace(0, 100000, 11)

def load_data(angle, alpha):
    folder = f"angle_{angle}"
    filename = f"results_angle_{angle}_auto_False_alpha_{alpha}_scale_1.0.npy"
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        # Assuming shape is (Num_Seeds, 11)
        return np.mean(np.load(filepath), axis=0)
    else:
        print(f"⚠️ Missing file: {filepath}")
        return None

def generate_separate_plots():
    for angle in ANGLES:
        plt.figure(figsize=(7, 5))
        
        best_alpha = None
        highest_return = -np.inf
        
        # Plot all alphas for this specific angle
        for alpha in ALPHAS:
            mean_curve = load_data(angle, alpha)
            
            if mean_curve is not None:
                plt.plot(STEPS, mean_curve, label=f'$\\alpha$ = {alpha}', lw=2.5)
                
                # Check the final steady-state return (average of last 3 points)
                final_return = np.mean(mean_curve[-3:])
                if final_return > highest_return:
                    highest_return = final_return
                    best_alpha = alpha
                    
        # Formatting the standalone plot
        plt.title(f'Sensitivity to Manual $\\alpha$ Tuning (Target Angle: {angle}°)\nOptimal $\\alpha_{{mnl}} \\approx {best_alpha}$', fontsize=11, fontweight='bold', pad=12)
        plt.xlabel('Environment Steps', fontsize=10)
        plt.ylabel('Average Return', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save as a distinct file
        filename = f'q5_manual_tuning_{angle}.png'
        plt.savefig(filename, dpi=300)
        plt.close() # Close figure to prevent overlap
        
        print(f"💾 Saved standalone figure: {filename}")

if __name__ == '__main__':
    generate_separate_plots()
    print("\n✅ All 4 plots generated successfully!")