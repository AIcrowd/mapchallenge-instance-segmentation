#!/bin/bash
set -e
# Train MMDetection instance segmentation model

# Default environment variables
GPU_DEVICES="3,4,5,6"  # Default to using 8 GPUs
# NUM_GPUS will be derived from GPU_DEVICES after argument parsing
CONFIG_DIR="./projects/testmap"  # Directory containing config files
CONFIG_FILE=""  # Will be set via command line
DATA_ROOT="/mapchallenge-instance-segmentation/mapping-challenge-v2.0/"  # Main data directory


function show_help {
  echo "Usage: $0 <CONFIG_FILE> [--gpus <GPU_DEVICES>] [--data-root <PATH>]"
  echo ""
  echo "Options:"
  echo "  --gpus         Comma-separated list of GPU device IDs to use (e.g., \"0,1\")"
  echo "                 The number of GPUs will be inferred from this list."
  echo "  --data-root    Path to dataset root directory"
  echo "  --list-configs List all available configurations"
  echo "  -h, --help     Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 xlarge_config.py"
  echo "  $0 xlarge_config.py --gpus 0,1,2,3"
  echo "  $0                   (runs all configs sequentially)"
}

function list_configs {
  echo "Available configurations in $CONFIG_DIR:"
  echo "----------------------------------------"
  find "$CONFIG_DIR" -name "*.py" -type f | sort | while read -r config; do
    echo "  $(basename "$config")"
  done
}

function run_training {
  local config_path="$1"
  local config_name="$(basename "$config_path")"
  
  # Distributed training parameters (defaults from user example)
  # These can be overridden by setting them as environment variables before running the script
  local NNODES=${NNODES:-1}
  local NODE_RANK=${NODE_RANK:-0}
  local PORT=${PORT:-29500} 
  local MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

  # Derive experiment name from config file (strip extension)
  local experiment_name="${config_name%.*}"

  # Set up work directory
  local work_dir="work_dirs/${experiment_name}"
  mkdir -p "$work_dir"

  # Log file
  local log_file="${work_dir}/training_$(date +'%Y%m%d_%H%M%S').log"
  # Construct annotation file paths
  local train_ann_file="${DATA_ROOT}train/annotations/annotation_non_augmented.json"
  local train_data_prefix_img="${DATA_ROOT}train/images/"
  local val_ann_file="${DATA_ROOT}val/annotations/annotation_non_augmented.json"
  local val_data_prefix_img="${DATA_ROOT}val/images/"
  local test_ann_file="${DATA_ROOT}test/annotations/annotation_non_augmented.json"
  local test_data_prefix_img="${DATA_ROOT}test/images/"

  echo "====================================================="
  echo "Starting distributed training with config: $config_name"
  echo "Using $NUM_GPUS GPUs: $GPU_DEVICES" # NUM_GPUS is now set globally
  echo "Work directory: $work_dir"
  echo "Log file: $log_file"
  echo "Master Addr: $MASTER_ADDR, Master Port: $PORT, NNODES: $NNODES, NODE_RANK: $NODE_RANK"
  echo "====================================================="

  # Construct arguments for tools/train.py
  # All custom key=value pairs are passed via --cfg-options
  local cfg_options_str="data_root=$DATA_ROOT \
train_dataloader.dataset.ann_file=$train_ann_file \
train_dataloader.dataset.data_prefix.img=$train_data_prefix_img \
val_dataloader.dataset.ann_file=$val_ann_file \
val_dataloader.dataset.data_prefix.img=$val_data_prefix_img \
test_dataloader.dataset.ann_file=$test_ann_file \
test_dataloader.dataset.data_prefix.img=$test_data_prefix_img \
val_evaluator.ann_file=$val_ann_file \
test_evaluator.ann_file=$test_ann_file"

  local train_py_args=(
      "$config_path"
      --work-dir "$work_dir"
      --launcher pytorch # Crucial for MMDetection with torch.distributed.launch
  )
  if [[ -n "$cfg_options_str" ]]; then
      train_py_args+=(--cfg-options $cfg_options_str)
  fi
  
  # Run the distributed training
  # Assumes this script is run from the root of the MMDetection project.
  # PYTHONPATH=".:$PYTHONPATH" ensures that Python can find modules in the current directory.
  CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
  PYTHONPATH=".:$PYTHONPATH" \
  python3 -m torch.distributed.launch \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --nproc_per_node=$NUM_GPUS \
      --master_port=$PORT \
      tools/train.py \
      "${train_py_args[@]}" \
      | tee "$log_file"
      
  echo "Completed training for: $config_name"
  echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --list-configs)
      list_configs
      exit 0
      ;;
    --gpus)
      GPU_DEVICES="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    *)
      if [[ -z "$CONFIG_FILE" ]]; then # Only take the first positional argument as CONFIG_FILE
        CONFIG_FILE="$1"
      else
        echo "Warning: Ignoring extra positional argument '$1'"
      fi
      shift
      ;;
  esac
done

# Calculate NUM_GPUS from GPU_DEVICES after parsing arguments
if [[ -z "$GPU_DEVICES" ]]; then
  echo "Error: GPU_DEVICES is not set. Please specify GPUs with --gpus or ensure a default is set."
  show_help
  exit 1
fi
IFS=',' read -ra GPUS_ARRAY <<< "$GPU_DEVICES"
NUM_GPUS=${#GPUS_ARRAY[@]}

if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "Error: No GPUs specified (NUM_GPUS is 0 based on GPU_DEVICES: '$GPU_DEVICES'). Training requires at least one GPU."
  exit 1
fi

# If config file is provided, run only that one
if [[ -n "$CONFIG_FILE" ]]; then
  FULL_CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  if [[ ! -f "$FULL_CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $FULL_CONFIG_PATH"
    echo "Available configurations:"
    list_configs
    exit 1
  fi
  
  run_training "$FULL_CONFIG_PATH"
else
  # No config specified, run all configs sequentially
  echo "No specific config provided. Running all configurations sequentially."
  echo ""
  
  # Get all config files
  config_files=$(find "$CONFIG_DIR" -name "*.py" -type f | sort)
  
  if [[ -z "$config_files" ]]; then
    echo "No configuration files found in $CONFIG_DIR"
    exit 1
  fi
  
  # Run each config file
  echo "Found $(echo "$config_files" | wc -l) configuration files."
  echo "Starting sequential training..."
  echo ""
  
  for config in $config_files; do
    run_training "$config"
  done
  
  echo "All training runs completed!"
fi
