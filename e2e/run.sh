#!/bin/bash

#SBATCH --job-name=E2E-CNN
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=m24oc-s1852485
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2

source /work/m24oc/m24oc/s1852485/Diss/train/e2e/virt_env/bin/activate
cd $SLURM_SUBMIT_DIR
export MPLCONFIGDIR="/work/m24oc/m24oc/s1852485/Diss/.matplotlib"
export HOME="/work/m24oc/m24oc/s1852485/Diss/.fakehome"
export PYTORCH_KERNEL_CACHE_PATH="/work/m24oc/m24oc/s1852485/Diss/.pykernelcache"

mkdir -p $MPLCONFIGDIR
mkdir -p $HOME
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

srun --ntasks-per-node=2 python train.py
