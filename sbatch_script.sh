#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=r18d512
#SBATCH --partition=gpu
#SBATCH --mem=40G
#SBATCH --output=r18d512.%j.out
#SBATCH --gres=gpu:v100-sxm2:1
module load cuda
module load anaconda3/3.7
source activate pytorch_env
python pre_train.py --knn -a resnet18 --dataset cifar10 --epochs 100 --results-dir cifar10 --moco-dim 512