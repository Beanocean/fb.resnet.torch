#!/bin/bash

# @Mail:   beanocean@outlook.com
# @D&T:    Wed 20 Jun 2018 03:40:45 PM AEST

export LD_LIBRARY_PATH="$HOME/.local/opt/cudnn-8.0-v5.1/lib64:/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
ROOT="/home/lg/workspace/tools/fb.resnet.torch/pretrained"
# indir=$1
# outdir=$2
# GPU=0

# th $ROOT/extfeat_for_dataset.lua $ROOT/../checkpoints/resnet-200.t7 20 $indir $outdir
# CUDA_VISIBLE_DEVICES=$GPU th $ROOT/extfeat_frame_folder.lua $ROOT/../checkpoints/resnet-200.t7 20 $indir $outdir
# CUDA_VISIBLE_DEVICES=$GPU th $ROOT/extfeat_flist_npy.lua $ROOT/../checkpoints/resnet-200.t7 20 $indir $outdir

outdir='/home/lg/remote/data/MSRVTT/2017/feat/resnet200@4'
[ ! -e $outdir ] && mkdir -p $outdir

for x in `seq 0 3`; do
  inlist='/home/lg/remote/data/MSRVTT/2017/tmp/x0'
  CUDA_VISIBLE_DEVICES=$x nohup th $ROOT/extfeat_flist_npy.lua $ROOT/../checkpoints/resnet-200.t7 20 "${inlist}${x}" $outdir > ${inlist}${x}.log 2>&1 &
done

wait
echo "Finish !"
