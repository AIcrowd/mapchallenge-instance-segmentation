#!/bin/bash

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./:$PYTHONPATH

# Function to run training
run_training() {
    local config_path=$1
    local num_gpus=${2:-8}

    echo "Training with config: $config_path"
    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
        tools/train.py $config_path --launcher pytorch
}

# Create log directory
mkdir -p logs

# Run experiments
run_training projects/mapchallenge/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py > logs/mask2former_swin-s.log 2>&1
run_training projects/mapchallenge/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco.py > logs/queryinst_r101.log 2>&1
run_training projects/mapchallenge/queryinst_r50_fpn_ms-2x_coco.py > logs/queryinst_r50.log 2>&1
run_training projects/mapchallenge/rtmdet-ins_X_8xb16-300e_satellite.py > logs/rtmdet-ins_X.log 2>&1
run_training projects/mapchallenge/rtmdet-ins_m_8xb32-300e_satellite.py > logs/rtmdet-ins_m.log 2>&1

echo "All experiments completed. Check logs in the logs/ and workdirs/ directory."
