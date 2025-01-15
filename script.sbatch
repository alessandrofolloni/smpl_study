#!/bin/bash
#SBATCH --job-name=train3dSMPLX
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessandro.folloni2@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=/public.hpc/alessandro.folloni2/logs/smpl_study/%x.o
#SBATCH --error=/public.hpc/alessandro.folloni2/logs/smpl_study/%x.e
#SBATCH --gres=gpu:1

. /public.hpc/alessandro.folloni2/smpl_venv/bin/activate
cd /public.hpc/alessandro.folloni2/smpl_study || exit
python3 joints3D_to_SMPLX/train_3D_to_SMPLX.py
