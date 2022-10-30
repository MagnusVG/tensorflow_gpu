#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=python_tensorflow_setup
#SBATCH --output=tensorflow_setup.out
 
# Set up environment
uenv verbose cuda-10.0 cudnn-10.0-7.6.5
uenv miniconda-python39
conda create -n tensorflow_env -c tensorflow-gpu numpy pandas matplotlib scikit-learn imbalanced-learn -y