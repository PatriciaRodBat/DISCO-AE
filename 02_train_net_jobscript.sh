#!/bin/bash

#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

module load Python/3.10
module load CUDA/11.7

#export DNNL_MAX_CPU_ISA=AVX2
#export MKL_ENABLE_INSTRUCTIONS=AVX2

source venvPatri3.10/bin/activate
cd /home/s33mrodr/DISCO-AE

echo "Start training"

echo "Running on $(hostname)"
nvidia-smi

python 02_train_network.py --config gallop