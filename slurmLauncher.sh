#!/bin/bash
CORES=2
NEIGHBORS=5
#SBATCH -N 1 
#SBATCH -p RM
#SBATCH --ntasks-per-node 2
#SBATCH -t 3:00:00 # ~ 52:00:00 (HH:MM:SS) for unparallelized
#SBATCH --mail-user=jmschabdach@gmail.com

# echo commands to stdout
set -x

# move to working directory on pylon1
cd /pylon1/ms4s88p/jms565

# copy files to working directory on pylon1
mkdir COPDGene_pickleFiles 
cp /pylon2/ms4s88p/jms565/code/COPDGene_pickleFiles/* ./COPDGene_pickleFiles/
mkdir test_results/

# need to copy code, too?
cp /home/jms565/code/buildGraphSingleSubjectCy.py .

# run code
for i in {0..7191} ; do
  python buildGraphSingleSubjectCy.py -n "${CORES}" -k "${NEIGHBORS}" -c "${i}"
done

# copy output file back to my directory on pylon2
cp test_results/* /pylon2/ms4s88p/jms565/graph_results/
