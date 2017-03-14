#!/bin/bash
#SBATCH -J build-graphs
#SBATCH -N 1 
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node 2
#SBATCH -t 0:20:00 # HH:MM:SS
#SBATCH -o /pylon1/ms4s88p/jms565/stdout-logs/task_%A_%a.out
#SBATCH -e /pylon1/ms4s88p/jms565/stderr-logs/task_%A_%a.out

# echo commands to stdout
set -x

echo /home/jms565/code/compileGraph.py -c $CORES -s ${SLURM_ARRAY_TASK_ID}

/home/jms565/code/compileGraph.py -c $CORES -s ${SLURM_ARRAY_TASK_ID}