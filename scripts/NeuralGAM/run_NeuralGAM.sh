#!/bin/sh

#SBATCH -J ngam-google #job name
#SBATCH -n 16 # number of tasks (numero de procesadores)
#SBATCH --mail-type=begin #Envía un correo cuando el trabajo inicia
#SBATCH --mail-type=end #Envía un correo cuando el trabajo finaliza
#SBATCH --mail-user=iortega@gradiant.org #Dirección a la que se envía
#SBATCH --mem=16G
#SBATCH -o logs/NeuralGAM-%A-%a.out #specify stdout output file (%A expands to array jobID, %a expands to array task id)

echo "Setting up environment...."
module load python

source ./env/bin/activate

echo "Starting iteration $SLURM_ARRAY_TASK_ID"

echo "Starting training..."
python main.py logistic -d uniform -o results -c 0.00001 -a 0.1 -i $SLURM_ARRAY_TASK_ID 
echo "Starting test..."
echo "Done iteration $SLURM_ARRAY_TASK_ID"
