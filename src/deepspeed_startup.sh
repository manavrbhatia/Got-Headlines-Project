#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=595FinalDeepspeed
#SBATCH --partition=spgpu
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f22_class
#SBATCH --time=50:00:00
#SBATCH --output=%x-output.log

module load python/3.10.4
module load cuda/11.6.2
source ../venv/bin/activate
deepspeed --num_gpus=4 main.py pegx_ally > deep_out.txt
