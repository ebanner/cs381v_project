#!/usr/local/bin/bash

# added by Anaconda2 2.4.0 installer
export PATH="/u/ebanner/.anaconda2/bin:$PATH"

source activate py27

cuda=/opt/cuda-7.5
cuDNN=/u/ebanner/builds/cudnn-7.0-linux-x64-v3.0-prod

export LD_LIBRARY_PATH=$cuDNN/lib64:$cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$cuDNN/include:$CPATH
export LIBRARY_PATH=$cuDNN/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=$cuDNN

export CUDA_HOME=$cuda

python train.py $@

source deactivate
