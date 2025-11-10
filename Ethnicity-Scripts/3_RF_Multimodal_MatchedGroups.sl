#!/bin/bash -e

#SBATCH --job-name=RF_Multimodal_MatchedGroups
#SBATCH --time=100:00:00
#SBATCH --mem=30GB
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
export PYTHONUNBUFFERED=1
python ./RF_Multimodal_MatchedGroups.py "${SLURM_ARRAY_TASK_ID}"
