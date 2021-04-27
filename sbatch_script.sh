#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --job-name=65768
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --output=k65768.%j.out
#SBATCH --gres=gpu:v100-sxm2:1
module load cuda
module load anaconda3/3.7
source activate pytorch_env
python pre_train.py --knn -a resnet50 --epochs 100 --results-dir cifar10 --dataset cifar10 --moco-k 65768
