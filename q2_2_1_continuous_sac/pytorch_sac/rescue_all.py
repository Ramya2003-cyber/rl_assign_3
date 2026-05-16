import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def ultimate_rescue():
    output_base = "RECOVERED_DATA"
    os.makedirs(output_base, exist_ok=True)
    
    print("--- 🕵️ Deep Search Started ---")
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if "events.out.tfevents" in file:
                path = os.path.join(root, file)
                print(f"\nProcessing: {path}")
                
                try:
                    ea = EventAccumulator(path)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    
                    for tag in tags:
                        data = ea.Scalars(tag)
                        df = pd.DataFrame(data)
                        df = df.rename(columns={'step': 'step', 'value': 'reward'})
                        
                        clean_tag = tag.replace('/', '_')
                        unique_id = root.replace('/', '_').replace('.', 'root')
                        
                        filename = f"{unique_id}_{clean_tag}.csv"
                        df.to_csv(os.path.join(output_base, filename), index=False)
                        print(f"   ✅ Saved {tag} -> {filename}")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not read file: {e}")

    print(f"\n🎉 ALL DONE! All your data is now in the '{output_base}' folder.")
    print("Check that folder in VSCode and you will see CSVs for every seed.")

if __name__ == "__main__":
    ultimate_rescue()
