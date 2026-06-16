#!/bin/bash
# Run MAE pre-training
python -m src.pretrain.mae_pretrain --data_2d data/processed/2d_1024px --data_3d data/processed/3d_224px --epochs 50 --batch_size 32