#!/bin/bash
python="/public.hpc/alessandro.folloni2/smpl_venv/bin/python3.9"
n_gpu=${GPU:-1}
if [[ $1 == "--version" ]] || [[ $1 == "-V" ]]; then
    $python "$1"
elif [[ $@ == *"generator3.py"* ]] || [[ $@ == *"import socket"* ]] || [[ $n_gpu == 0 ]] || [[ $@ == *"packaging_tool.py"* ]]; then
    $python "$@"
else
    /usr/bin/srun -Q --immediate=60 --gres=gpu:$n_gpu $python "$@"
fi
