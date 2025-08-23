#!/bin/bash
#SBATCH -J PLR_Political_Missing
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=06:00:00
#SBATCH --array=0-0

module load slurm_setup
source ~/.conda_init
conda activate viktor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

srun python3 -m MORE.experiments_gb.PLR_main_missingLabels
