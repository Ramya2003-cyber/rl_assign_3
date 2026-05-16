#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

for i in {1..15}
do
   echo "Starting Hover Auto Seed $i"
   python train.py \
     experiment=hover_auto \
     seed=$i \
     agent.params.learnable_temperature=true \
     replay_buffer_capacity=100000 \
     num_train_steps=600000
done
