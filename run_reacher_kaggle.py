import subprocess
import os

# 1. Force MuJoCo headless rendering
custom_env = os.environ.copy()
custom_env["MUJOCO_GL"] = "egl"

# 2. The exact same 15 seeds for all three reward formulations
seeds = [10, 42, 100, 256, 512, 1024, 2048, 4096, 8192, 12345, 54321, 99999, 11111, 22222, 33333]

# 3. CHANGE THIS FOR EACH KAGGLE NOTEBOOK ('a', 'b', or 'c')
REWARD_TYPE = 'c' 

for seed in seeds:
    # --- NEW CHECK: Does the results file already exist? ---
    expected_file = f"reacher_results_{REWARD_TYPE}_seed_{seed}.npy"
    if os.path.exists(expected_file):
        print(f"⏭️ SKIPPING SEED {seed}: '{expected_file}' already exists!")
        continue
    # --------------------------------------------------------

    print(f"\n=======================================================")
    print(f"🚀 STARTING REACHER | REWARD {REWARD_TYPE.upper()} | SEED {seed}")
    print(f"=======================================================\n")
    
    cmd = [
        "python", "run_reacher.py", 
        f"reward_type={REWARD_TYPE}", 
        f"seed={seed}",
        "num_train_steps=500000"
    ]
    
    # Run the command with the headless rendering environment variable
    subprocess.run(cmd, env=custom_env, check=True)

print("\n🎉 ALL 15 SEEDS COMPLETE! DOWNLOAD YOUR .NPY FILES! 🎉")