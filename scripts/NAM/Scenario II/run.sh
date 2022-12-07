#!/bin/sh

#SBATCH -J ngam-google #job name
#SBATCH -n 16 # number of tasks (numero de procesadores)
#SBATCH --mail-type=begin #Envía un correo cuando el trabajo inicia
#SBATCH --mail-type=end #Envía un correo cuando el trabajo finaliza
#SBATCH --mail-user=iortega@gradiant.org #Dirección a la que se envía
#SBATCH --mem=16G
#SBATCH -o logs/nam-deep-googleDataset-%A-%a.out #specify stdout output file (%A expands to array jobID, %a expands to array task id)

echo "Setting up environment...."
module load python

source ./env/bin/activate


for i in {1..1000}
do
    echo "Starting iteration $i"
    # USAGE: nam_train_google_deep.py [-i INPUT] [-o OUTPUT] [-f Distribution Family. Use gaussian for LINEAR REGRESSION and binomial for LOGISTIC REGRESSION] [-t ITERATION]
    python nam_train_google_deep.py -i ./dataset/google -o results-nam-deep-exu -f gaussian -t $i 
    python test_nam_google_deep.py -i ./dataset/google -o results-nam-deep-exu -f gaussian -t $i

    rm -r results-nam-deep-exu/$i/google/model/
    echo "Done iteration $i"

done