#!/bin/bash

nvidia-smi

module list

module av

# CUDA 환경변수 확인
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# PyTorch CUDA 설정 확인
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"