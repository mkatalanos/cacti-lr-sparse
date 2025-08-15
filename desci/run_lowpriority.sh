#!/bin/bash

#SBATCH --job-name=lr-DESCI-tuning
#SBATCH --time=48:00:00
#SBATCH --partition=standard
#SBATCH --qos=lowpriority
#SBATCH --account=m24oc-s1852485
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
## SBATCH --exclusive
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

module load openmpi/
module load intel-20.4/compilers
module load gcc/10.2.0

source /work/m24oc/m24oc/s1852485/Diss/repo/desci/virt_env/bin/activate

cd $SLURM_SUBMIT_DIR

START=$SLURM_ARRAY_TASK_ID
STEP=100
TOTAL=216

LINE=$(( START + 1 ))
while [ $LINE -le $TOTAL ]; do
  PARAMS=$(sed -n "${LINE}p" params.txt)
  srun python automation.py $PARAMS
  # echo $LINE
  LINE=$(( LINE + STEP ))
done


