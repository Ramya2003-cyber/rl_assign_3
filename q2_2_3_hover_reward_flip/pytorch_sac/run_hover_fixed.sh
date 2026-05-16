#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

for i in {1..15}
do
   echo "Starting Hover Fixed Seed $i"
   python train.py experiment=hover_fixed seed=$i agent.params.learnable_temperature=false agent.params.init_temperature=0.01 replay_buffer_capacity=100000 num_train_steps=600000
done
