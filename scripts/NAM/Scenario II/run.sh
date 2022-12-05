#!/bin/sh

#SBATCH -J ngam-google #job name
#SBATCH -n 16 # number of tasks (numero de procesadores)
#SBATCH --mail-type=begin #Envía un correo cuando el trabajo inicia
#SBATCH --mail-type=end #Envía un correo cuando el trabajo finaliza
#SBATCH --mail-user=iortega@gradiant.org #Dirección a la que se envía
#SBATCH --mem=16G
#SBATCH -o logs/nam-exu-ortegaSestelo-1024-%A-%a.out #specify stdout output file (%A expands to array jobID, %a expands to array task id)

echo "Setting up environment...."
module load python

source ./env/bin/activate

echo "Starting iteration $SLURM_ARRAY_TASK_ID"

python nam_train_google_deep.py logistic -d uniform -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID
python test_nam_google_deep.py logistic -d uniform -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID

rm -r results-nam-deep-exu/$SLURM_ARRAY_TASK_ID/uniform_binomial/model/

python nam_train_google_deep.py linear -d uniform -t homoscedastic -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID
python test_nam_google_deep.py linear -d uniform -t homoscedastic -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID

rm -r results-nam-deep-exu/$SLURM_ARRAY_TASK_ID/homoscedastic_uniform_gaussian/model/

python nam_train_google_deep.py linear -d uniform -t heteroscedastic -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID
python test_nam_google_deep.py linear -d uniform -t heteroscedastic -o results-nam-deep-exu -i $SLURM_ARRAY_TASK_ID

rm -r results-nam-deep-exu/$SLURM_ARRAY_TASK_ID/heteroscedastic_uniform_gaussian/model/

echo "Done iteration $SLURM_ARRAY_TASK_ID"
