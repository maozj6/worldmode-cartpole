#!/bin/bash
#SBATCH --job-name=trainLargeMDN    # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem-per-cpu=16000mb
#SBATCH --time=24:05:00               # Time limit hrs:min:sec
#SBATCH --output=gpu_task_%j.log   # Standard output error log
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --error=gpu_task_%j.err

module load conda
conda activate tea
python -u train-large-mdn.py --train="/train/" --test="/test/"


date
