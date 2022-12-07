#!/bin/sh

#SBATCH -J NeuralGAM_Scenario_I_R2 #job name
#SBATCH -n 16 # number of tasks (numero de procesadores)
#SBATCH --mail-type=begin #Envía un correo cuando el trabajo inicia
#SBATCH --mail-type=end #Envía un correo cuando el trabajo finaliza
#SBATCH --mail-user=iortega@gradiant.org #Dirección a la que se envía
#SBATCH --mem=16G
#SBATCH -o logs/NeuralGAM_Scenario_I_R2.out #specify stdout output file (%A expands to array jobID, %a expands to array task id)

echo "Setting up environment...."
module load python

source ./env/bin/activate

for i in {1..1000}
do
    echo "Starting iteration $i"
    python main_simulation.py linear -d uniform -t heteroscedastic -i $i -c 10e-5 -a 0.1 -u 1024 -o results_NeuralGAM_Scenario_I
    echo "Done iteration $i"
done

echo "DONE!"