#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu20 
#SBATCH --time=168:00:00
#SBATCH --job-name=preprocess_area
#SBATCH --output=preprocess_area_part1.out
 
# Activate environment
uenv miniconda-python39
conda activate pandas_env
# Run the Python script that uses the GPU
python -u preprocess_data.py --part=1 --neighbors=2