#!/bin/bash

#SBATCH --job-name=DESCI-Measurements
#SBATCH --time=10:00:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=22
#SBATCH --cpus-per-task=1


#SBATCH --account=m24oc-s1852485
#SBATCH --partition=standard
#SBATCH --qos=lowpriority

module load matlab

export HOME=/work/m24oc/m24oc/s1852485/Diss/.fakehome/

matlab_wrapper -nodisplay < /work/m24oc/m24oc/s1852485/Diss/DeSCI/test_desci_custom.m

