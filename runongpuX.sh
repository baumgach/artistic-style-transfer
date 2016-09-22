#!/bin/bash

echo "Running $2 on GPU$1"

CUDA_VISIBLE_DEVICES=''
CUDA_VISIBLE_DEVICES=$1; export CUDA_VISIBLE_DEVICES

THEANO_FLAGS=floatX=float32,device=gpu0,force_device=True PATH=${PATH}:/vol/cuda/7.5.18/bin/ LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/7.5.18/lib64 CPATH=/vol/cuda/7.5.18/lib64/include LIBRARY_PATH=/vol/cuda/7.5.18/lib64 python $2


