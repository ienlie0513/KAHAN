#!/bin/bash

#SBATCH --job-name parallel   ## name that will show up in the queue
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=3  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --partition=normal  ## the partitions to run in (comma seperated)
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)

# Execute job steps
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "sleep 10; echo 'hello 1'" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "sleep 20; echo 'hello 2'" &
srun --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "sleep 30; echo 'hello 3'" &
wait