import os
import numpy as np

def patch_files():
    print(f"{'='*50}\n🛠️  STARTING STEP 0 SURGERY\n{'='*50}")

    # 1. Gather the healthy Step 0 data from the Auto-Tuned files
    donor_files = {}
    for root, _, files in os.walk('.'):
        for file in files:
            if 'auto_True' in file and file.endswith('.npy'):
                filepath = os.path.join(root, file)
                try:
                    data = np.load(filepath)
                    # If it's a healthy 11-step file, harvest its 0th column
                    if len(data.shape) == 2 and data.shape[1] == 11:
                        angle = file.split('_')[2]
                        donor_files[angle] = data[:, 0] 
                except:
                    continue

    if not donor_files:
        print("❌ CRITICAL: Could not find any 11-step 'auto_True' files to harvest Step 0 data from.")
        return

    print(f"🧬 Harvested Step 0 DNA for angles: {list(donor_files.keys())}\n")

    # 2. Patch all incomplete 10-step files
    patched_count = 0
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.npy'):
                filepath = os.path.join(root, file)
                try:
                    data = np.load(filepath)
                    
                    # Target files that only have 10 evaluation columns
                    if len(data.shape) == 2 and data.shape[1] == 10:
                        angle = file.split('_')[2]

                        if angle in donor_files:
                            donor_col = donor_files[angle]
                            num_seeds = data.shape[0]

                            # Slice the donor column to match the exact number of seeds 
                            # (This handles that one file that stopped at 12 seeds!)
                            step_0_data = donor_col[:num_seeds].reshape(-1, 1)

                            # Stitch Step 0 to the front of the 10-step array
                            patched_data = np.hstack((step_0_data, data))

                            # Overwrite the file with the new 11-step array
                            np.save(filepath, patched_data)
                            print(f"✅ Patched: {file} -> Now {patched_data.shape}")
                            patched_count += 1
                        else:
                            print(f"⚠️ Missing donor data for angle {angle}, skipping {file}")
                except:
                    continue

    print(f"\n🏁 SURGERY COMPLETE: Successfully patched {patched_count} files!")

if __name__ == '__main__':
    patch_files()