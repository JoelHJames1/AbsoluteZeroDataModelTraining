#!/bin/bash
# Author: Joel Hernandez James
# Current Date: 2025-05-11
# Description: Script to run the AZR training system

# Set up environment
set -e  # Exit on error

# Display banner
echo "======================================================"
echo "  Absolute Zero Reasoner (AZR) Training System"
echo "  Author: Joel Hernandez James"
echo "  Date: 2025-05-11"
echo "======================================================"
echo ""

# Check for Python 3.10+
python_version=$(python3 --version | cut -d' ' -f2)
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required (found $python_version)"
    echo "Please install Python 3.10+ and try again"
    exit 1
fi

echo "Python version: $python_version ✓"

# Check for required dependencies
echo "Checking dependencies..."

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  $1 ✓"
    else
        echo "  $1 ✗ (missing)"
        missing_deps=1
    fi
}

missing_deps=0
check_package torch
check_package transformers
check_package accelerate
check_package bitsandbytes
check_package datasets
check_package wandb
check_package trl
check_package yaml
check_package tqdm

if [ $missing_deps -eq 1 ]; then
    echo ""
    echo "Some dependencies are missing. Install them with:"
    echo "pip install torch transformers accelerate bitsandbytes datasets wandb trl pyyaml tqdm"
    
    read -p "Would you like to install the missing dependencies now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch transformers accelerate bitsandbytes datasets wandb trl pyyaml tqdm
    else
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
fi

# Create necessary directories
echo "Setting up directory structure..."
mkdir -p data logs models

# Parse command line arguments
RESUME=""
CONFIG="config/azr_config.yaml"
EVAL_ONLY=0
BENCHMARK="humaneval"
MAX_TASKS=0

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --resume)
            RESUME="$2"
            shift
            shift
            ;;
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --eval-only)
            EVAL_ONLY=1
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
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run.sh [--resume CHECKPOINT] [--config CONFIG_PATH] [--eval-only] [--benchmark BENCHMARK] [--max-tasks N]"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Using config file: $CONFIG"

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/azr_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run in evaluation mode if requested
if [ $EVAL_ONLY -eq 1 ]; then
    echo "Running in evaluation mode..."
    echo "Benchmark: $BENCHMARK"
    
    if [ -z "$RESUME" ]; then
        echo "Error: Model checkpoint must be specified with --resume when using --eval-only"
        exit 1
    fi
    
    MAX_TASKS_ARG=""
    if [ $MAX_TASKS -gt 0 ]; then
        MAX_TASKS_ARG="--max_tasks $MAX_TASKS"
    fi
    
    echo "Evaluating model: $RESUME"
    cd scripts
    python evaluation.py --model_path "../$RESUME" --benchmark $BENCHMARK --config "../$CONFIG" $MAX_TASKS_ARG | tee -a "../$LOG_FILE"
    cd ..
    
    echo ""
    echo "Evaluation complete. Results saved to logs/evaluation_results.json"
    exit 0
fi

# Run training
echo "Starting AZR training..."

if [ -n "$RESUME" ]; then
    echo "Resuming from checkpoint: $RESUME"
    cd scripts
    python train.py --config "../$CONFIG" --resume "../$RESUME" | tee -a "../$LOG_FILE"
else
    echo "Starting new training run"
    cd scripts
    python train.py --config "../$CONFIG" | tee -a "../$LOG_FILE"
fi

echo ""
echo "Training complete!"
echo "Logs saved to: $LOG_FILE"
echo ""
echo "To evaluate the trained model, run:"
echo "./run.sh --eval-only --resume models/checkpoint-XXXX --benchmark humaneval"
