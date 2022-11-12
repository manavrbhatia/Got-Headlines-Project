#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=595EvaluateModel
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f22_class
#SBATCH --time=0:30:00
#SBATCH --output=%x-output.log

module load python/3.10.4
module load cuda
source ../venv/bin/activate

python3 evaluate.py generic_allyears input.txt > eval_result.out
