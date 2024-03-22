#!/bin/bash
#SBATCH -J PLR-Political-Missing
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
# 256 is the maximum reasonable value for CooLMUC-3
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=02:00:00
module load slurm_setup

source ~/.conda_init
conda activate viktor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python PLR_politicalDataSet_Missing.py

