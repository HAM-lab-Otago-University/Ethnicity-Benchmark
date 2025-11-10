#!/bin/bash -e

#SBATCH --job-name=pls_abccCntr_sim_Ewa
#SBATCH --time=20:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=10
#SBATCH --account=uoo03493
#SBATCH --profile task
#SBATCH --output=/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/log/%x/%x_%j_%a.out

#SBATCH --mail-user=lalfa602@student.otago.ac.nz
#SBATCH --mail-type=ALL

#SBATCH --array=0,2


module purge
module load Python/3.10.5-gimkl-2022a
source /nesi/project/uoo03493/PLS_venv/bin/activate

export PYTHONNOUSERSITE=1

python ./pls_abccCntr_sim_Ewa.py "${SLURM_ARRAY_TASK_ID}"
