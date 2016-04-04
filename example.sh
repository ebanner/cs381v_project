#!/bin/bash

# Minimal script to run keras code on gpu on maverick interactively

cuda=/opt/apps/cuda/7.5
cuDNN=/work/03859/ebanner/maverick/builds/cudnn-7.0-linux-x64-v3.0-prod

export LD_LIBRARY_PATH=$cuDNN/lib64:$cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$cuDNN/include:$CPATH
export LIBRARY_PATH=$cuDNN/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=$cuDNN

export CUDA_HOME=$cuda

python train.py $*
