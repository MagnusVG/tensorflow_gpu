#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=24:00:00
#SBATCH --job-name=tensorflow_model
#SBATCH --output=tensorflow_model_c05_n2.out
 
# Activate environment
uenv verbose cuda-10.0 cudnn-10.0-7.6.5
uenv miniconda-python39
conda activate tensorflow_env
# Run the Python script that uses the GPU
python -u neural_network.py --folder="cells_05/neighbors_2/" --batchSize=32 --model="c05_n2"