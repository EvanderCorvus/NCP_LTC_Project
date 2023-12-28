#!/bin/bash --login
# this line tells that this is a bash script.
 
# The following commands are arguments to PBS:
#PBS -N z_NCP_LTC
#PBS -S /bin/bash
#PBS -l select=ncpus=1:mem=5gb:host=Imest002
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -r y
 
# This line specifies the working directory of the submitted job:
cd $PBS_O_WORKDIR
 
# Needed for python code only:
source /usr/local/python3/bin/activate

python3 main.py