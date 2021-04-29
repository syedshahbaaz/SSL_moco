#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=r50cifar
#SBATCH --partition=gpu
#SBATCH --mem=60G
#SBATCH --output=r50cifar.%j.out
#SBATCH --gres=gpu:v100-sxm2:1
module load cuda
module load anaconda3/3.7
source activate pytorch_env
python pre_train.py --knn -a resnet50 --dataset cifar10 --epochs 100 --results-dir cifar10 --moco-k 32768 &
sleep 60
python pre_train.py --knn -a resnet50 --dataset cifar10 --epochs 100 --results-dir cifar10 --moco-k 65536
