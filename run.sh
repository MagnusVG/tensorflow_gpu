#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=12:00:00
#SBATCH --job-name=tensorflow_model
#SBATCH --output=tensorflow_model_01.out
 
# Activate environment
uenv verbose cuda-10.0 cudnn-10.0-7.6.5
uenv miniconda-python39
conda activate tensorflow_env
# Run the Python script that uses the GPU
python -u model_test.py