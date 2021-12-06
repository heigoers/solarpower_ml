#!/bin/bash
 
#SBATCH --job-name metSat 
#SBATCH --mem 64000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 8:20:00

module purge
module load tensorflow2/py3.cuda10.0

#python autoencoderCluster.py  > log_ae.out
python trainModel.py > log.out
