# Absolute Zero Reasoner (AZR) Model Training

## Overview

This project implements the Absolute Zero Reasoner (AZR) approach for training a Large Language Model (LLM) to excel at coding tasks. Based on the paper [Absolute Zero: Learning Reasoning from First Principles](https://www.arxiv.org/pdf/2505.03335), this system enables a model to teach itself through a self-play mechanism inspired by AlphaZero.

## Key Features

- **Self-Improving AI**: The model generates its own training data and learns from its successes and failures
- **Zero-Shot Learning**: No human-labeled data required for training
- **Benchmark Tracking**: Real-time tracking of performance against industry-standard benchmarks
- **Modern Dashboard**: Interactive React-based dashboard for monitoring training progress
- **Optimized for Apple Silicon**: Runs efficiently on MacBook Pro M3 Max with 48GB RAM

## Architecture

The AZR system consists of several key components:

1. **Task Proposer**: Generates programming tasks of varying difficulty levels
2. **Task Solver**: Attempts to solve the generated tasks using the current model
3. **Executor**: Safely executes and validates solutions
4. **Trainer**: Updates the model using reinforcement learning based on task outcomes
5. **Evaluator**: Measures performance against standard benchmarks
6. **Dashboard**: Visualizes training progress and model performance

## Technical Details

- **Base Model**: Qwen3-4B (quantized to 8-bit for efficiency)
- **Training Method**: Proximal Policy Optimization (PPO)
- **Parameter-Efficient Fine-Tuning**: Using LoRA for efficient adaptation
- **Benchmarks**: HumanEval, MBPP, and APPS

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Node.js and npm (for the React dashboard)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/joelhjames21/azr-model-training.git
cd azr-model-training
```

2. Install Python dependencies:
```bash
pip install torch transformers accelerate bitsandbytes datasets wandb trl peft
```

3. Install React dashboard dependencies:
```bash
cd azr/dashboard-react
npm install
cd ../..
```

### Usage

#### Start Training with Dashboard

To start training with the modern React dashboard:
```bash
cd azr
./run_training_with_dashboard.sh
```

This will:
1. Start the AZR training process
2. Start the Flask API server on port 5000
3. Start the React development server on port 3000

You can customize the training with options:
```bash
cd azr
./run_training_with_dashboard.sh --steps 20000 --config custom_config.yaml
```

Options:
- `--steps <number>`: Number of training steps (default: 10000)
- `--config <path>`: Path to configuration file (default: config/azr_config.yaml)
- `--resume <path>`: Path to checkpoint to resume from

#### Dashboard Only

If you just want to run the dashboard without training:
```bash
cd azr
./run_dashboard.sh
```

#### Evaluation

To evaluate the model on benchmarks:
```bash
cd azr
./run.sh --eval-only --benchmark humaneval
```

## Project Structure

```
azr-model-training/
└── azr/
    ├── api/                   # Flask API for React dashboard
    ├── config/                # Configuration files
    ├── dashboard/             # Simple dashboard
    ├── dashboard-react/       # Modern React dashboard
    ├── data/                  # Training data and logs
    ├── logs/                  # Log files
    ├── models/                # Model checkpoints
    └── scripts/               # Core scripts
        ├── evaluation.py      # Benchmark evaluation
        ├── executor.py        # Safe code execution
        ├── proposer.py        # Task generation
        ├── solver.py          # Task solving
        ├── train.py           # Main training loop
        └── utils.py           # Utility functions
```

## Performance

The AZR approach has shown promising results in improving coding capabilities:

- **HumanEval**: Starting from ~20% and targeting 80%+ pass@1
- **MBPP**: Starting from ~25% and targeting 80%+ pass@1
- **APPS**: Starting from ~15% and targeting 40%+ pass@1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The authors of the Absolute Zero paper for the innovative approach
- The Qwen team for the base model
- The Hugging Face team for the transformers library
