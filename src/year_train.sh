#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# CHANGE THE YEAR FOR THESE!

# Set up batch job settings
#SBATCH --job-name=595Practical2016Pipeline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --account=eecs595f22_class
#SBATCH --time=16:00:00
#SBATCH --output=year_results/2016/%x-output.log

module load python/3.10.4
module load cuda
source ../venv/bin/activate

python3 practical_year_model.py 2016 > year_results/2016/2016_bench.out
#python3 practical_year_model.py 2017 > year_results/2017/2017_bench.out
#python3 practical_year_model.py 2018 > year_results/2018/2018_bench.out
#python3 practical_year_model.py 2019 > year_results/2019/2019_bench.out
#python3 practical_year_model.py 2020 > year_results/2020/2020_bench.out
