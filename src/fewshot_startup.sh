#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=595FewShotPegx
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f22_class
#SBATCH --time=16:00:00
#SBATCH --output=fewshot_results/pegx_ally/%x-output.log

module load python/3.10.4
module load cuda
source ../venv/bin/activate

python3 oneshot_eval.py pegx_ally > fewshot_results/pegx_ally/pegx_bench.out
