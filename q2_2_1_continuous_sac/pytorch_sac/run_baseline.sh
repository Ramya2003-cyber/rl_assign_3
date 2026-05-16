#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
for i in {1..15}
do
   python train.py env=LunarLanderContinuous-v3 seed=$i experiment=baseline num_train_steps=300000 log_save_tb=false
done
