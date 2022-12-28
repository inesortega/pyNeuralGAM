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

max=20
finish=$((max * SLURM_ARRAY_TASK_ID))
init=$((finish - max))
init=$((init + 1))

for i in $(seq $init $finish)
do
    echo "Starting iteration $i"
    python train_nam_Scenario_I_exu.py linear -d uniform -t homoscedastic -o results_nam_Scenario_I_exu -i $i
    python train_nam_Scenario_I_exu.py linear -d uniform -t homoscedastic -o results_nam_Scenario_I_exu -i $i
    rm -r results_nam_Scenario_I_exu/$i/homoscedastic_uniform_gaussian/model/

    python train_nam_Scenario_I_exu.py linear -d uniform -t heteroscedastic -o results_nam_Scenario_I_exu -i $i
    python test_nam_Scenario_I_exu.py linear -d uniform -t heteroscedastic -o results_nam_Scenario_I_exu -i $i
    rm -r results_nam_Scenario_I_exu/$i/heteroscedastic_uniform_gaussian/model/

    python train_nam_Scenario_I_exu.py logistic -d uniform -o results_nam_Scenario_I_exu -i $i
    python test_nam_Scenario_I_exu.py logistic -d uniform -o results_nam_Scenario_I_exu -i $i
    rm -r results_nam_Scenario_I_exu/$i/uniform_binomial/model/
    
    echo "Done iteration $i"

done

echo "DONE!"