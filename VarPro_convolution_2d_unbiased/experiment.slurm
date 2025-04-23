#!/bin/bash

#SBATCH -A gnf@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --job-name=VarPro_unbiased   # nom du job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=output_%x_%A_%a.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=output_%x_%A_%a.out   # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=32,128,512,1024      # indices des travaux

module purge
module load pytorch-gpu/py3/2.3.0

# echo des commandes lancées
set -x

# La valeur de ${SLURM_ARRAY_TASK_ID} est differente pour chaque travail.
srun python Experiment.py --lmbda 1e-4 --student_width ${SLURM_ARRAY_TASK_ID} --epochs 25000
srun python Experiment.py --lmbda 1e-2 --student_width ${SLURM_ARRAY_TASK_ID} --epochs 25000
srun python Experiment.py --lmbda 1e0 --student_width ${SLURM_ARRAY_TASK_ID} --epochs 25000
