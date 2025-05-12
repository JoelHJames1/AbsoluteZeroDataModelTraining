#!/bin/bash
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Script: run.sh

# Description:  
# Main entry point script for the AZR system

# Print header
echo "======================================================"
echo "  Absolute Zero Reasoner (AZR) Training System"
echo "  Author: Joel Hernandez James"
echo "  Date: $(date +%Y-%m-%d)"
echo "======================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION ✓"

# Check dependencies
echo "Checking dependencies..."
pip_install_if_missing() {
    if ! python -c "import $1" &> /dev/null; then
        echo "  $1 ✗ (installing...)"
        pip install $2
    else
        echo "  $1 ✓"
    fi
}

pip_install_if_missing "torch" "torch"
pip_install_if_missing "transformers" "transformers"
pip_install_if_missing "accelerate" "accelerate"
pip_install_if_missing "bitsandbytes" "bitsandbytes -U"
pip_install_if_missing "datasets" "datasets"
pip_install_if_missing "wandb" "wandb"
pip_install_if_missing "trl" "trl"
pip_install_if_missing "peft" "peft"
pip_install_if_missing "yaml" "pyyaml"

# Create necessary directories
mkdir -p logs data models

# Force update bitsandbytes to latest version
echo "Updating bitsandbytes to latest version..."
pip install -U bitsandbytes

# Parse command line arguments
DASHBOARD=false
DASHBOARD_ONLY=false
DASHBOARD_PORT=8080
TRAINING_PORT=8081
EVAL_ONLY=false
BENCHMARK=""
MAX_TASKS=0
CONFIG_PATH="config/azr_config.yaml"
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dashboard)
            DASHBOARD=true
            shift
            ;;
        --dashboard-only)
            DASHBOARD_ONLY=true
            shift
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift
            shift
            ;;
        --training-port)
            TRAINING_PORT="$2"
            shift
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift
            shift
            ;;
        --max-tasks)
            MAX_TASKS="$2"
            shift
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift
            shift
            ;;
        --resume)
            RESUME_PATH="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start dashboard if requested
if [ "$DASHBOARD" = true ] || [ "$DASHBOARD_ONLY" = true ]; then
    echo "Starting dashboard server on port $DASHBOARD_PORT..."
    python dashboard/server.py --port $DASHBOARD_PORT &
    DASHBOARD_PID=$!
    
    # Wait for dashboard to start
    sleep 2
    echo "Dashboard running at http://localhost:$DASHBOARD_PORT"
fi

# Exit if dashboard-only mode
if [ "$DASHBOARD_ONLY" = true ]; then
    echo "Running in dashboard-only mode. Connect to a running training process."
    wait $DASHBOARD_PID
    exit 0
fi

# Run evaluation if requested
if [ "$EVAL_ONLY" = true ]; then
    echo "Running evaluation..."
    EVAL_CMD="python scripts/evaluation.py"
    
    if [ ! -z "$RESUME_PATH" ]; then
        EVAL_CMD="$EVAL_CMD --model $RESUME_PATH"
    fi
    
    if [ ! -z "$BENCHMARK" ]; then
        EVAL_CMD="$EVAL_CMD --benchmark $BENCHMARK"
    fi
    
    if [ "$MAX_TASKS" -gt 0 ]; then
        EVAL_CMD="$EVAL_CMD --max-tasks $MAX_TASKS"
    fi
    
    $EVAL_CMD
    exit 0
fi

# Start training
echo "Starting AZR training..."
TRAIN_CMD="python scripts/train.py"

if [ ! -z "$CONFIG_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --config $CONFIG_PATH"
fi

if [ ! -z "$RESUME_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
fi

# Run training
$TRAIN_CMD

# Clean up
if [ "$DASHBOARD" = true ]; then
    kill $DASHBOARD_PID
fi

echo "AZR training completed."
