#!/bin/bash

# Launch two training tasks on CUDA 0 and two on CUDA 1 in parallel, each with its own nohup output file

CUDA_VISIBLE_DEVICES=0 nohup python launcher/examples/train_pretrain.py --seed 0 > nohup_pretrain_seed0.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python launcher/examples/train_pretrain.py --seed 1 > nohup_pretrain_seed1.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python launcher/examples/train_pretrain.py --seed 2 > nohup_pretrain_seed2.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python launcher/examples/train_pretrain.py --seed 3 > nohup_pretrain_seed3.out 2>&1 &

wait
echo "All training processes have completed."