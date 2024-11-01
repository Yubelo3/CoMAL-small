#!/bin/bash
#SBATCH --job-name=simple_job
#SBATCH --account=msccsit2024

# Enable email notificaitons when job begins and ends, uncomment if you need it
# #SBATCH --mail-user=yzhoufv@connect.ust.hk
# #SBATCH --mail-type=begin
# #SBATCH --mail-type=end

#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a30  # intel, amd or gpu-a30

python main_contrastive.py