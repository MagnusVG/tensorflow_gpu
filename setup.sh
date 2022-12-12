#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=tf_setup
#SBATCH --output=tf_setup.out
 
# Set up environment
uenv verbose cuda-10.0 cudnn-10.0-7.6.5
uenv miniconda-python39
conda create -n tensorflow_env -c numpy matplotlib pandas tensorflow-gpu scikit-learn pyarrow -y
conda install --name tensorflow_env -c https://conda.anaconda.org/conda-forge/ imbalanced-learn