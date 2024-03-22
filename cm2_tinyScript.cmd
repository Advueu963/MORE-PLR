#!/bin/bash
#SBATCH -J PLR-Clas-RF
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=56
# 56 is the maximum reasonable value for CooLMUC-2
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=48:00:00
module load slurm_setup

source ~/.conda_init
conda activate viktor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python PLR_main_randomForest.py
