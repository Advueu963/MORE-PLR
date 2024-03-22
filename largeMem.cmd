#!/bin/bash
#SBATCH -J EVALUATION_PLR_GPR
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --partition=cm2_inter_large_mem
#SBATCH --mail-type=end
#SBATCH --mem=80000MB
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=50:00:00
module load slurm_setup

source ~/.conda_init
conda activate viktor
export OMP_NUM_THREADS=40

srun python PLR_GaussinProcMultiOutput.py
