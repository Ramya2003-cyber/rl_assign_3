import numpy as np
import subprocess
import os

def run_experiment(angle, auto_tune, alpha=0.1, scale=1.0):
    cmd = [
        "python", "scripts/run_pendulum.py",
        f"custom.target_angle={angle}",
        f"custom.auto_tune={auto_tune}",
        f"custom.alpha={alpha}",
        f"custom.reward_scale={scale}"
    ]
    print(f"--- STARTING: Angle {angle} | Auto {auto_tune} | Alpha {alpha} | Scale {scale} ---")
    subprocess.run(cmd)

def get_best_alpha(angle, test_alphas):
    """Reads the generated .npy files and finds the alpha with the highest final return."""
    best_alpha = None
    best_return = -float('inf')

    for alpha in test_alphas:
        # This filename must perfectly match the one you defined in run_pendulum.py!
        filename = f"results_angle_{angle}_auto_False_alpha_{alpha}_scale_1.0.npy"
        
        if os.path.exists(filename):
            data = np.load(filename)
            # data shape is [15_seeds, 10_eval_points]. 
            # We want the average performance of the final evaluation point (index -1) across all 15 seeds.
            final_mean_return = np.mean(data[:, -1]) 
            
            print(f"Analysis: Alpha {alpha} scored {final_mean_return:.2f}")
            
            if final_mean_return > best_return:
                best_return = final_mean_return
                best_alpha = alpha
        else:
            print(f"Warning: Could not find file {filename}")
            
    return best_alpha

if __name__ == '__main__':
    # ... [Keep your Q2 automated tuning runs here] ...
    # q2_angles = [0,-10,-60,30,90, -90, 120, -150]
    # for angle in q2_angles:
    #     run_experiment(angle=angle, auto_tune="true")
    # ---------------------------------------------------------
    # QUESTION 5(a): The Manual Sweep
    # ---------------------------------------------------------
    q5_angles = [-60, 90, 120, -150]
    test_alphas = [0.01, 0.05, 0.1, 0.2, 0.5] # Added a few more for a thorough sweep
    
    for angle in q5_angles:
        for alpha in test_alphas:
            run_experiment(angle=angle, auto_tune="false", alpha=alpha)

    # ---------------------------------------------------------
    # THE AUTOMATION INTERCEPT
    # ---------------------------------------------------------
    # print("\n--- AUTOMATICALLY DETERMINING BEST MANUAL ALPHA FOR 90 DEGREES ---")
    # best_guess_manual_alpha = get_best_alpha(angle=90, test_alphas=test_alphas)
    # print(f"--- THE WINNER IS: Alpha {best_guess_manual_alpha} --- \n")

    # ---------------------------------------------------------
    # QUESTION 5(b): Reward Scaling (Using the Automated Winner)
    # ---------------------------------------------------------
    # scales = [10.0, 0.1]
    
    # # Fallback just in case something broke during the file read
    # if best_guess_manual_alpha is None:
    #     best_guess_manual_alpha = 0.1 
        
    # for scale in scales:
    #     # Auto-tuned scaling
    #     run_experiment(angle=90, auto_tune="true", scale=scale)
    #     # Manual-tuned scaling using the auto-selected best alpha!
    #     run_experiment(angle=90, auto_tune="false", alpha=best_guess_manual_alpha, scale=scale)

    print("ALL EXPERIMENTS COMPLETED. GOOD MORNING!")