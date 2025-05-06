#!/bin/bash

# Train MMDetection instance segmentation model

# Default environment variables
GPU_DEVICES="0,1,2,3,4,5,6,7"  # Default to using 8 GPUs
NUM_GPUS=8
CONFIG_DIR="./projects/mapchallenge"  # Directory containing config files
CONFIG_FILE=""  # Will be set via command line
DATA_ROOT="/instance_segmentation/"  # Main data directory

function show_help {
  echo "Usage: $0 <CONFIG_FILE> [--gpus <GPU_DEVICES>] [--data-root <PATH>]"
  echo ""
  echo "Options:"
  echo "  --gpus         Comma-separated list of GPU device IDs to use"
  echo "  --data-root    Path to dataset root directory"
  echo "  -h, --help     Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 xlarge_config.py"
  echo "  $0 xlarge_config.py --gpus 0,1,2,3"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
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
      CONFIG_FILE="$1"
      shift
      ;;
  esac
done

# Check if config file was provided
if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: No config file specified"
  show_help
  exit 1
fi

# Setup full config path
FULL_CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
if [[ ! -f "$FULL_CONFIG_PATH" ]]; then
  echo "Error: Config file not found: $FULL_CONFIG_PATH"
  exit 1
fi

# Derive experiment name from config file (strip extension)
EXPERIMENT_NAME="${CONFIG_FILE%.*}"

# Set up work directory
WORK_DIR="work_dirs/${EXPERIMENT_NAME}"
mkdir -p "$WORK_DIR"

# Log file
LOG_FILE="${WORK_DIR}/training_$(date +'%Y%m%d_%H%M%S').log"

# Run the training
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
python3 tools/train.py "$FULL_CONFIG_PATH" \
    --work-dir "$WORK_DIR" \
    --cfg-options data_root="$DATA_ROOT" \
    | tee "$LOG_FILE"
