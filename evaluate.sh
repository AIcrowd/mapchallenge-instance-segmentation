#!/bin/bash

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./:$PYTHONPATH

# Function to evaluate model
evaluate_model() {
    local config_path=$1
    local checkpoint_path=$2
    local output_file=$3

    echo "Evaluating model: $config_path"
    python tools/test.py $config_path $checkpoint_path \
        --format-only \
        --out $output_file
}


# Best model checkpoints (adjust paths as needed)
MASK2FORMER_SWIN_S_CKPT="work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/best_segm_mAP.pth"
MASK2FORMER_SWIN_L_CKPT="work_dirs/mask2former_swin-L_8xb32-24k_coco/best_segm_mAP.pth"
QUERYINST_R101_CKPT="work_dirs/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco/best_segm_mAP.pth"
QUERYINST_R50_CKPT="work_dirs/queryinst_r50_fpn_ms-2x_coco/best_segm_mAP.pth"
RTMDET_X_CKPT="work_dirs/rtmdet-ins_X_8xb16-300e_satellite/best_segm_mAP.pth"
RTMDET_M_CKPT="work_dirs/rtmdet-ins_m_8xb32-300e_satellite/best_segm_mAP.pth"

# Evaluate models
evaluate_model \
    projects/mapchallenge/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py \
    $MASK2FORMER_SWIN_S_CKPT
    
evaluate_model \
    projects/mapchallenge/mask2former_swin-L_8xb32-24k_coco.py \
    $MASK2FORMER_SWIN_L_CKPT
    
evaluate_model \
    projects/mapchallenge/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco.py \
    $QUERYINST_R101_CKPT 

evaluate_model \
    projects/mapchallenge/queryinst_r50_fpn_ms-2x_coco.py \
    $QUERYINST_R50_CKPT

evaluate_model \
    projects/mapchallenge/rtmdet-ins_X_8xb16-300e_satellite.py \
    $RTMDET_X_CKPT 

evaluate_model \
    projects/mapchallenge/rtmdet-ins_m_8xb32-300e_satellite.py \
    $RTMDET_M_CKPT 

