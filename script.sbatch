#!/bin/bash
#SBATCH --job-name=train2d3d
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessandro.folloni2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=/public.hpc/alessandro.folloni2/logs/smpl_study/%x.o
#SBATCH --error=/public.hpc/alessandro.folloni2/logs/smpl_study/%x.e
#SBATCH --gres=gpu:1

. /public.hpc/alessandro.folloni2/smpl_venv/bin/activate
cd /public.hpc/alessandro.folloni2/smpl_study || exit
python3 joints2D_to_3D/train_2D_to_3D.py
