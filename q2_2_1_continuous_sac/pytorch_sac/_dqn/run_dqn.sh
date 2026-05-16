#!/bin/bash
for i in {1..15}
do
   echo "Starting DQN Seed $i"
   python train.py --env LunarLander-v3 --seed $i --num_train_steps 300000
done