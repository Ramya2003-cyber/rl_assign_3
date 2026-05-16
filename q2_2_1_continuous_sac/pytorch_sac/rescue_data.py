import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def rescue_all_data():
    search_dir = "." 
    print(f"--- Starting Brute Force Search in {os.getcwd()} ---")
    
    found_any = False
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if "events.out.tfevents" in file:
                found_any = True
                path = os.path.join(root, file)
                print(f"\nFound Event File: {path}")
                
                ea = EventAccumulator(path)
                ea.Reload()
                
                tags = ea.Tags()['scalars']
                print(f"   Available Tags: {tags}")
                
                target_tag = None
                for t in ['eval/episode_reward', 'eval_episode_reward', 'eval/reward']:
                    if t in tags:
                        target_tag = t
                        break
                
                if target_tag:
                    data = ea.Scalars(target_tag)
                    df = pd.DataFrame(data)
                    df = df.rename(columns={'step': 'step', 'value': 'episode_reward'})
                    
                    folder_name = os.path.basename(root)
                    if folder_name == 'tb':
                        folder_name = os.path.basename(os.path.dirname(root))
                    
                    save_path = f"results_csv/{folder_name}"
                    os.makedirs(save_path, exist_ok=True)
                    df.to_csv(os.path.join(save_path, 'eval.csv'), index=False)
                    print(f"   ✅ SUCCESS: Saved to {save_path}/eval.csv")
                else:
                    print(f"   ❌ No reward tags found in this file.")

    if not found_any:
        print("No Tensorboard files found at all. Check your folder structure!")

if __name__ == "__main__":
    rescue_all_data()
