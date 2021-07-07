#!/bin/bash

source activate ganspace
cd /mnt/share/hao/scripts/weathering/stylegan2-pytorch

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node=10 --master_port=6331 train.py --batch 8 /mnt/share/hao/datasets/weathering/lmdb