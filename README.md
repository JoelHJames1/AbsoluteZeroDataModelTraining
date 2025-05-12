# Absolute Zero Reasoner (AZR) Model Training

## Overview

This project implements the Absolute Zero Reasoner (AZR), a self-learning approach to training language models for code generation. Inspired by AlphaZero's self-play methodology, AZR generates its own training data through a continuous loop of task creation, solution, and validation.

The system is designed to run on consumer hardware (specifically a MacBook Pro M3 Max with 48GB RAM) by leveraging 4-bit quantization of the Qwen3-4B model.

The real-time dashboard provides visualization of training progress, benchmark performance, and task success rates.

## Key Features

- **Self-Learning Architecture**: Generates and solves its own programming tasks
- **Real-Time Dashboard**: Visualizes training progress and benchmark performance
- **Memory-Efficient**: Optimized for running on consumer hardware
- **Benchmark Tracking**: Continuously evaluates against HumanEval, MBPP, and APPS
- **Adaptive Difficulty**: Progressively increases task complexity as the model improves

## Architecture

The AZR system consists of several key components:

1. **Task Proposer**: Generates increasingly difficult programming tasks
2. **Task Solver**: Attempts to solve the generated tasks
3. **Executor**: Validates solutions by executing the code
4. **Trainer**: Implements the reinforcement learning loop
5. **Evaluator**: Measures performance against standard benchmarks
6. **Dashboard**: Provides real-time visualization of training progress

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Transformers
- Accelerate
- BitsAndBytes
- TRL (Transformer Reinforcement Learning)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/azr-model-training.git
cd azr-model-training

# Install dependencies
pip install torch transformers accelerate bitsandbytes datasets wandb trl

# Make the run script executable
chmod +x azr/run.sh
```

### Running the System

To start training with the dashboard:

```bash
cd azr
./run.sh --dashboard
```

For more options and detailed usage instructions, see the [AZR README](azr/README.md).

## How It Works

### Self-Learning Approach

AZR implements a reinforcement learning loop:

1. **Task Generation**: The model proposes programming tasks of varying difficulty
2. **Solution Attempt**: The model tries to solve its own tasks
3. **Validation**: Solutions are executed to verify correctness
4. **Reinforcement**: The model is updated based on solution success
5. **Benchmark Evaluation**: Performance is regularly measured against standard benchmarks

### Real-Time Monitoring

The dashboard provides live visualization of:

- Training progress and metrics
- Task success rates
- Benchmark performance compared to leading models
- Recent tasks and solutions

## Benchmarks

AZR aims to surpass state-of-the-art models on these coding benchmarks:

- **HumanEval**: Code generation from natural language descriptions
- **MBPP**: Python programming tasks
- **APPS**: Algorithmic problem-solving

## Project Structure

```
.
├── README.md                  # This file
└── azr/                       # Main AZR implementation
    ├── config/                # Configuration files
    ├── dashboard/             # Real-time visualization
    ├── data/                  # Training data and logs
    ├── logs/                  # Log files
    ├── models/                # Saved model checkpoints
    ├── scripts/               # Core implementation
    ├── README.md              # Detailed AZR documentation
    └── run.sh                 # Main entry point script
```

## Technical Details

### Model

- Base model: Qwen3-4B
- Quantization: 4-bit (using BitsAndBytes)
- Training: PPO (Proximal Policy Optimization)

### Hardware Requirements

- Minimum: 16GB RAM, modern CPU
- Recommended: 48GB RAM, Apple Silicon M3 or equivalent
- GPU: Optional but recommended for faster training

## License

[MIT License](LICENSE)

## Citation

```
@software{azr2025,
  author = {Hernandez James, Joel},
  title = {Absolute Zero Reasoner: Self-Learning Approach for Code Generation},
  year = {2025},
  url = {https://github.com/yourusername/azr-model-training}
}
```

## Acknowledgments

- The AlphaZero paper for inspiration on self-play methodology
- Hugging Face for the Transformers library
- The Qwen team for the base model
