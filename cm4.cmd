#!/bin/bash
#SBATCH -J PLR_EPSI_0.2_MISSING_0.6_CHAIN
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=05:00:00
#SBATCH --array=0-0

module load slurm_setup
source ~/.conda_init
conda activate viktor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

srun python3 -m MORE.experiments_gb.PLR_main_missingLabels_epsilon2
