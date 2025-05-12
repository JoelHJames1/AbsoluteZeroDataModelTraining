#!/bin/bash
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Script: run_training_with_dashboard.sh

# Description:  
# Script to run the AZR training process with the dashboard

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

# Set up trap to kill background processes on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Change to the script's directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p logs data models

# Parse command line arguments
STEPS=10000
CONFIG="config/azr_config.yaml"
RESUME=""

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Start the training process
echo "Starting AZR training process..."
if [ -n "$RESUME" ]; then
  python3 scripts/train.py --config "$CONFIG" --steps "$STEPS" --resume "$RESUME" &
else
  python3 scripts/train.py --config "$CONFIG" --steps "$STEPS" &
fi
TRAIN_PID=$!

# Wait for the training data server to start
echo "Waiting for training data server to start..."
sleep 5

# Start the dashboard
echo "Starting dashboard..."
./run_dashboard.sh &
DASHBOARD_PID=$!

# Wait for both processes
echo "Training and dashboard are running!"
echo "Training process: PID $TRAIN_PID"
echo "Dashboard: PID $DASHBOARD_PID"
echo "Press Ctrl+C to stop both processes."

wait $TRAIN_PID $DASHBOARD_PID
