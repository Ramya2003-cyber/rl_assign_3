import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import argparse
from dqn_agent import QNetwork, ReplayBuffer

def evaluate(env_name, policy_net, device, n_episodes=20):
    eval_env = gym.make(env_name)
    avg_reward = 0
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_t).argmax().item()
            state, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            avg_reward += reward
    eval_env.close()
    return avg_reward / n_episodes

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)
    
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(100000)
    
    log_dir = f"exp_local/{args.env}/dqn_seed_{args.seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    eval_results = []
    train_results = []
    
    step = 0
    episode = 0
    eps = 1.0
    next_eval_step = 0

    while step < args.num_train_steps:
        if step >= next_eval_step:
            print(f"\n>>> EVALUATING AT STEP {step}...")
            score = evaluate(args.env, policy_net, device)
            eval_results.append({'step': step, 'episode_reward': score})
            pd.DataFrame(eval_results).to_csv(f"{log_dir}/eval.csv", index=False)
            print(f">>> EVAL FINISHED. REWARD: {score:.2f}\n")
            next_eval_step += 10000

        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done and step < args.num_train_steps:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = policy_net(state_t).argmax().item()
            
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            memory.push(state, action, reward, next_state, float(term))
            state = next_state
            episode_reward += reward
            step += 1
            
            if len(memory) >= 64:
                s, a, r, ns, d = memory.sample(64)
                s_t, ns_t = torch.FloatTensor(s).to(device), torch.FloatTensor(ns).to(device)
                a_t, r_t, d_t = torch.LongTensor(a).to(device), torch.FloatTensor(r).to(device), torch.FloatTensor(d).to(device)
                
                curr_q = policy_net(s_t).gather(1, a_t.unsqueeze(1))
                with torch.no_grad():
                    max_next_q = target_net(ns_t).max(1)[0]
                    target_q = r_t + (0.99 * max_next_q * (1 - d_t))
                
                loss = F.mse_loss(curr_q, target_q.unsqueeze(1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if step % 100 == 0:
                print(f"Step: {step:6d} | Ep: {episode:4d} | Eps: {eps:.2f} | R: {episode_reward:7.2f}", end="\r")

        train_results.append({'step': step, 'episode_reward': episode_reward})
        pd.DataFrame(train_results).to_csv(f"{log_dir}/train.csv", index=False)

        eps = max(eps * 0.995, 0.05)
        episode += 1
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=300000)
    
    args = parser.parse_args()
    
    train(args)
