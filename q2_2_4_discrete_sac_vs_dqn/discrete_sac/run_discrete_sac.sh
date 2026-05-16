#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

for i in {1..15}
do
   echo "Starting Discrete SAC Seed $i"
   python train.py \
     env=LunarLander-v3 \
     experiment=discrete_sac \
     seed=$i \
     num_train_steps=300000 \
     num_seed_steps=10000 \
     num_eval_episodes=20
done