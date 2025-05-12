# Absolute Zero Reasoner (AZR)

## Overview

The Absolute Zero Reasoner (AZR) is a self-learning system that trains language models to excel at coding tasks through a reinforcement learning approach. Inspired by the AlphaZero methodology, AZR generates its own training data through self-play, continuously improving its performance on programming tasks.

This implementation uses the Qwen3-4B model with 4-bit quantization, optimized for running on Apple Silicon hardware with limited memory (48GB RAM).

## Key Features

- **Self-Play Training**: AZR generates its own programming tasks and solutions, creating a continuous learning loop
- **Adaptive Difficulty**: Task difficulty increases as the model improves
- **Real-Time Dashboard**: Monitor training progress, benchmark performance, and task success rates
- **Benchmark Evaluation**: Regularly evaluates against standard coding benchmarks (HumanEval, MBPP, APPS)
- **Memory-Efficient**: Uses 4-bit quantization to run efficiently on consumer hardware

## System Architecture

```
azr/
├── config/                # Configuration files
│   └── azr_config.yaml    # Main configuration
├── dashboard/             # Real-time training visualization
│   ├── index.html         # Dashboard UI
│   ├── styles.css         # Dashboard styling
│   ├── dashboard.js       # Dashboard frontend logic
│   └── server.py          # Dashboard backend server
├── data/                  # Training data and logs
├── logs/                  # Log files
├── models/                # Saved model checkpoints
├── scripts/               # Core implementation
│   ├── executor.py        # Code execution and validation
│   ├── proposer.py        # Task generation
│   ├── solver.py          # Task solving
│   ├── train.py           # Main training loop
│   ├── evaluation.py      # Benchmark evaluation
│   └── utils.py           # Utility functions
└── run.sh                 # Main entry point script
```

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Accelerate
- BitsAndBytes
- TRL (Transformer Reinforcement Learning)
- Datasets
- Weights & Biases (optional for tracking)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/azr.git
   cd azr
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers accelerate bitsandbytes datasets wandb trl
   ```

3. Make the run script executable:
   ```bash
   chmod +x run.sh
   ```

## Usage

### Starting Training with Dashboard

To start training with the real-time dashboard:

```bash
./run.sh --dashboard
```

This will:
1. Start the dashboard server on port 8080
2. Start the training process with real-time data streaming
3. Open the dashboard in your default browser

### Dashboard-Only Mode

To run just the dashboard (connecting to an already running training process):

```bash
./run.sh --dashboard-only
```

### Training Options

```bash
./run.sh [OPTIONS]

Options:
  --resume CHECKPOINT      Resume training from a checkpoint
  --config CONFIG_PATH     Use a custom configuration file
  --dashboard              Enable the real-time dashboard
  --dashboard-only         Run only the dashboard
  --dashboard-port PORT    Set custom dashboard port (default: 8080)
  --training-port PORT     Set custom training data port (default: 8081)
  --eval-only              Run evaluation only
  --benchmark BENCHMARK    Specify benchmark for evaluation
  --max-tasks N            Limit number of tasks for evaluation
```

### Evaluation

To evaluate a trained model on benchmarks:

```bash
./run.sh --eval-only --resume models/checkpoint-XXXX --benchmark humaneval
```

## Benchmarks

AZR tracks performance against these coding benchmarks:

- **HumanEval**: Evaluates code generation from natural language descriptions
- **MBPP (Mostly Basic Python Programming)**: Tests Python programming skills
- **APPS**: Assesses algorithmic problem-solving abilities

The system aims to surpass state-of-the-art models like GPT-3.5, Claude, and CodeLlama.

## Dashboard

The real-time dashboard provides:

- Training status and progress
- Success rate and reward metrics
- Task difficulty progression
- Benchmark performance comparison
- Recent tasks with solutions

## Configuration

Edit `config/azr_config.yaml` to customize:

- Model parameters
- Training hyperparameters
- Task generation settings
- Evaluation settings
- Benchmark targets

## How It Works

1. **Task Generation**: The system generates programming tasks of increasing difficulty
2. **Task Solving**: The model attempts to solve these tasks
3. **Validation**: Solutions are executed and validated
4. **Reinforcement Learning**: The model is updated based on solution success
5. **Benchmark Evaluation**: Periodically evaluated against standard benchmarks
6. **Visualization**: Progress is displayed on the real-time dashboard

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@software{azr2025,
  author = {Hernandez James, Joel},
  title = {Absolute Zero Reasoner: Self-Learning Approach for Code Generation},
  year = {2025},
  url = {https://github.com/yourusername/azr}
}
