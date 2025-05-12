# Absolute Zero Reasoner (AZR)

## Overview

The Absolute Zero Reasoner (AZR) is a self-improving AI system that learns to solve programming tasks through a process inspired by AlphaZero. It uses a combination of self-play, reinforcement learning, and large language models to continuously improve its problem-solving abilities.

## Architecture

AZR follows the Absolute Zero approach described in the paper [Absolute Zero: Learning Reasoning from First Principles](https://www.arxiv.org/pdf/2505.03335). The system consists of the following components:

- **Task Proposer**: Generates programming tasks of varying difficulty
- **Task Solver**: Attempts to solve the generated tasks
- **Executor**: Validates solutions by executing them in a safe environment
- **Trainer**: Updates the model using reinforcement learning based on task outcomes
- **Evaluator**: Measures performance against standard benchmarks

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers library
- Accelerate
- BitsAndBytes (for quantization)
- TRL (for reinforcement learning)
- PEFT (for parameter-efficient fine-tuning)

## Setup

1. Install dependencies:
```bash
pip install torch transformers accelerate bitsandbytes datasets wandb trl peft
```

2. Configure the system in `config/azr_config.yaml`

## Usage

### Training

To start training the model:

```bash
./run.sh
```

Options:
- `--config <path>`: Path to configuration file (default: `config/azr_config.yaml`)
- `--resume <path>`: Path to checkpoint to resume from
- `--dashboard`: Start the training dashboard
- `--dashboard-port <port>`: Dashboard port (default: 8080)

### Evaluation

To evaluate the model on benchmarks:

```bash
./run.sh --eval-only --benchmark <benchmark_name>
```

Supported benchmarks:
- `humaneval`: HumanEval benchmark for code generation
- `mbpp`: MBPP benchmark for Python programming
- `apps`: APPS benchmark for problem-solving

### Dashboard

AZR includes two dashboard options:

#### 1. Simple Dashboard

The simple dashboard is included in the main training script:

```bash
./run.sh --dashboard
```

#### 2. React Dashboard (Modern UI)

The React dashboard provides a more modern and interactive interface:

```bash
./run_dashboard.sh
```

This will start:
- Flask API server on port 5000
- React development server on port 3000

The React dashboard connects to the training process in real-time and displays:
- Training status and metrics
- Real-time task generation and solutions
- Performance charts
- Benchmark progress

## Project Structure

```
azr/
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

## Benchmarks

AZR tracks progress against the following benchmarks:

- **HumanEval**: Measures code generation capabilities
- **MBPP**: Measures Python programming abilities
- **APPS**: Measures problem-solving and algorithmic thinking

## License

This project is licensed under the MIT License - see the LICENSE file for details.
