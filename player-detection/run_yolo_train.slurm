#!/bin/bash
#SBATCH --job-name=torch-t1-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-22:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#Set up environment
source ~/software/init-conda
conda activate pytorch

#Call your script
python ~/final_project/basketball_detection.py

#Deactivate conda
conda deactivate

echo "Job finished at $(date)"
