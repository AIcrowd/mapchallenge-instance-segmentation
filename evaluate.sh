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
        --eval segm \
        --out $output_file
}

# Create results directory
mkdir -p results

# Best model checkpoints (adjust paths as needed)
MASK2FORMER_SWIN_S_CKPT="work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/best_segm_mAP.pth"
QUERYINST_R101_CKPT="work_dirs/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco/best_segm_mAP.pth"
QUERYINST_R50_CKPT="work_dirs/queryinst_r50_fpn_ms-2x_coco/best_segm_mAP.pth"
RTMDET_X_CKPT="work_dirs/rtmdet-ins_X_8xb16-300e_satellite/best_segm_mAP.pth"
RTMDET_M_CKPT="work_dirs/rtmdet-ins_m_8xb32-300e_satellite/best_segm_mAP.pth"

# Evaluate models
evaluate_model \
    projects/mapchallenge/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py \
    $MASK2FORMER_SWIN_S_CKPT \
    results/mask2former_swin-s_results.json

evaluate_model \
    projects/mapchallenge/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco.py \
    $QUERYINST_R101_CKPT \
    results/queryinst_r101_results.json

evaluate_model \
    projects/mapchallenge/queryinst_r50_fpn_ms-2x_coco.py \
    $QUERYINST_R50_CKPT \
    results/queryinst_r50_results.json

evaluate_model \
    projects/mapchallenge/rtmdet-ins_X_8xb16-300e_satellite.py \
    $RTMDET_X_CKPT \
    results/rtmdet-ins_X_results.json

evaluate_model \
    projects/mapchallenge/rtmdet-ins_m_8xb32-300e_satellite.py \
    $RTMDET_M_CKPT \
    results/rtmdet-ins_m_results.json

# Generate comprehensive evaluation summary
python <<EOF
import json

results_files = [
    'results/mask2former_swin-s_results.json',
    'results/queryinst_r101_results.json',
    'results/queryinst_r50_results.json',
    'results/rtmdet-ins_X_results.json',
    'results/rtmdet-ins_m_results.json'
]

summary = {}
for file in results_files:
    with open(file, 'r') as f:
        data = json.load(f)
        model_name = file.split('_results')[0].split('/')[-1]
        summary[model_name] = data['segm_mAP']

print("Evaluation Summary:")
for model, map_score in summary.items():
    print(f"{model}: segm mAP = {map_score}")

with open('results/evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
EOF

echo "Evaluation completed. Check results in the results/ directory."
