#!/bin/bash
 
#SBATCH --job-name metSat_2a
#SBATCH --mem 128000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 10:00:00

#module purge
#module load tensorflow2/py3.cuda10.0

python 2a_trainModel_keras.py > 2a_log.out
