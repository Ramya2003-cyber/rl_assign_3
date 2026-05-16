#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

for i in {1..15}
do
   echo "Starting Seed $i"
   python train.py env=LunarLanderContinuous-v3 seed=$i experiment=baseline_seed_$i
done
