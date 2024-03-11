#!/bin/bash
#SBATCH --job-name=trainCartVAE    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
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
python -u train_large.py --train="train/" --test="test/"


date
