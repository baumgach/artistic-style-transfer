#!/bin/bash

echo "Running $2 on GPU$1"

CUDA_VISIBLE_DEVICES=''
CUDA_VISIBLE_DEVICES=$1; export CUDA_VISIBLE_DEVICES

THEANO_FLAGS=floatX=float32,device=gpu0,force_device=True,base_compile_dir=/homes/cbaumgar/.theano2 PATH=${PATH}:/vol/cuda/7.5.18/bin/ LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/7.5.18/lib64:/vol/biomedic/users/cbaumgar/software/cuDNN/cuda/lib64 CPATH=/vol/biomedic/users/cbaumgar/software/cuDNN/cuda/include/ LIBRARY_PATH=/vol/biomedic/users/cbaumgar/software/cuDNN/cuda/lib64/ python $2


