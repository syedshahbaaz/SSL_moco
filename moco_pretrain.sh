#!/bin/bash

module load cuda/11.0
module load anaconda3/3.7

conda activate pytorch_env
python train.py *
conda deactivate