#!/bin/sh

#SBATCH -J nam_Scenario_I_R1 #job name
#SBATCH -n 16 # number of tasks (numero de procesadores)
#SBATCH --mail-type=begin #Envía un correo cuando el trabajo inicia
#SBATCH --mail-type=end #Envía un correo cuando el trabajo finaliza
#SBATCH --mail-user=iortega@gradiant.org #Dirección a la que se envía
#SBATCH --mem=16G
#SBATCH -o logs/nam_Scenario_I_R1.out #specify stdout output file (%A expands to array jobID, %a expands to array task id)

echo "Setting up environment...."
module load python

source ./env/bin/activate

for i in {1..1000}
do
    echo "Starting iteration $i"
    python train_nam_Scenario_I.py linear -d uniform -t homoscedastic -o results_nam_Scenario_I -i $i
    python test_nam_Scenario_I.py linear -d uniform -t homoscedastic -o results_nam_Scenario_I -i $i
    rm -r results_nam_Scenario_I/$i/homoscedastic_uniform_gaussian/model/
    echo "Done iteration $i"

done

echo "DONE!"