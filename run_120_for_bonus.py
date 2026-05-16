import subprocess

TARGET_ANGLE = 120
BUDGET = 1000
TOTAL_SEEDS = 15

print(f"🚀 STARTING LOCAL BATCH: Angle {TARGET_ANGLE}° | Budget {BUDGET}")
print("====================================================")

for seed in range(TOTAL_SEEDS):
    print(f"\n▶️ Running Seed {seed}/14...")
    
    cmd = [
        "python", "run_pebble.py",
        f"custom.target_angle={TARGET_ANGLE}",
        f"custom.feedback_budget={BUDGET}",
        f"seed={seed}",
        "num_train_steps=50000",   # Stops the 1-million-step infinite run
        "device=cuda"              # Forces PyTorch to use your RTX 4050
    ]
    
    try:
        # check=True stops the loop if a seed crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ CRASH DETECTED ON SEED {seed}. Stopping batch.")
        break

print("\n🎉 ====================================================")
print(f"ALL 15 SEEDS FOR {TARGET_ANGLE}° (Budget {BUDGET}) COMPLETE!")
print("=======================================================")