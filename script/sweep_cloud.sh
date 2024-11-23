#!/bin/bash
module load cuda/11.8

############## home/pljh0906 되어있는 경로는 전부 본인 anaconda 경로로 수정해야 함

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/pljh0906/anaconda3/envs/mvsplat/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1

export TORCH_CUDA_MEMORY_ALLOCATOR=native
export TORCH_USE_CUDA_DSA=1
CONDA_PATH=${1:-"$HOME/anaconda3"}

source $CONDA_PATH/etc/profile.d/conda.sh
conda activate mvsplat

export WANDB_DIR="/home/pljh0906/MVSplat_custom/wandb"
mkdir -p $WANDB_DIR

cd /home/pljh0906/MVSplat_custom

CONDA_PYTHON_PATH=$(which python)

export PYTHONPATH="/home/pljh0906/MVSplat_custom:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

nvidia-smi
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.backends.cudnn.version())"

echo "Running sweep agent with ID"
### Change the project name and entity to generated after wandb sweep command

### Example: wandb: Run sweep agent with: wandb agent pljh0906/yaicon-3-5gs/<project id>
### Run $CONDA_PYTHON_PATH -m wandb agent pljh0906/yaicon-3-5gs/<project id> ~~~
$CONDA_PYTHON_PATH -m wandb agent pljh0906/yaicon-3-5gs/0x1czxlj --project yaicon-3-5gs --entity pljh0906 --count 100