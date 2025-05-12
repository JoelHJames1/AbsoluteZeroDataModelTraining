#!/bin/bash
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Script: run_training_only.sh

# Description:  
# Script to run only the AZR training process without the dashboard

# Check if the script is run with sudo
if [ "$EUID" -eq 0 ]; then
  echo "Please do not run this script with sudo or as root."
  exit 1
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists python3; then
  echo "Python 3 is not installed. Please install Python 3 to run the training."
  exit 1
fi

# Change to the script's directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p logs data models

# Parse command line arguments
STEPS=100
CONFIG="config/azr_config.yaml"
RESUME=""
UPDATE_TRANSFORMERS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --update-transformers)
      UPDATE_TRANSFORMERS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Update Transformers if requested
if [ "$UPDATE_TRANSFORMERS" = true ]; then
  echo "Updating Transformers library..."
  pip install --upgrade transformers
  
  echo "Updating bitsandbytes library..."
  pip install --upgrade bitsandbytes
  
  echo "Do you want to install Transformers from source for the latest features? (y/n)"
  read -r response
  if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Installing Transformers from source..."
    pip install git+https://github.com/huggingface/transformers.git
  fi
fi

# Start the training process
echo "Starting AZR training process..."
if [ -n "$RESUME" ]; then
  python3 scripts/train.py --config "$CONFIG" --steps "$STEPS" --resume "$RESUME"
else
  python3 scripts/train.py --config "$CONFIG" --steps "$STEPS"
fi

echo "Training completed!"
