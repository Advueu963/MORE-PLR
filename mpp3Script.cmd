#!/bin/bash
#SBATCH -J <Job-Name>
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=flat,quad # use all memory equal
#SBATCH --mail-type=end
#SBATCH --mail-user=<E-MAIL>
#SBATCH --export=NONE
#SBATCH --time=05:00:00

module load slurm_setup
source ~/.conda_init
conda activate <ENVIROMENT>
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

srun python LR_main_missingLabels.py
