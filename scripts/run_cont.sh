echo "Starting execution of 1000 iterations"

sbatch -a 1-99 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 100-199 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 200-299 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 300-399 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 400-499 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 500-599 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 600-699 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 700-799 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 800-899 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 900-999 -t 03:00:00 run.sh &
sleep 1h

sbatch -a 1000-1000 -t 03:00:00 run.sh &

echo "Done!"
