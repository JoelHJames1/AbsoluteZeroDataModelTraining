# Absolute Zero Reasoner (AZR)

## Overview

The Absolute Zero Reasoner (AZR) is a self-improving AI system for code generation and problem-solving. Inspired by the AlphaZero approach, AZR uses reinforcement learning to train language models through self-play, continuously generating and solving programming tasks of increasing difficulty.

**Author:** Joel Hernandez James  
**Date:** 2025-05-11

## Architecture

AZR follows a modular architecture with the following components:

1. **Task Proposer**: Generates programming tasks of varying difficulty
2. **Task Solver**: Solves tasks using the language model
3. **Executor**: Safely executes and validates code solutions
4. **Trainer**: Implements the reinforcement learning loop
5. **Evaluator**: Benchmarks model performance against standard coding tasks

## System Requirements

- Python 3.10+
- MacOS (optimized for Apple Silicon M3 Max)
- 48GB+ RAM
- CUDA-compatible GPU (optional, for faster training)

## Installation

### Dependencies

```bash
# Install system dependencies
brew install cmake protobuf rust python@3.10

# Install Python dependencies
pip install torch transformers accelerate bitsandbytes datasets wandb trl
```

### Setup

1. Clone this repository:
```bash
git clone https://github.com/joelhjames21/azr.git
cd azr
```

2. Create necessary directories (if not using the provided scripts):
```bash
mkdir -p data logs models
```

3. Configure Weights & Biases (optional but recommended for tracking):
```bash
wandb login
```

## Configuration

The system is configured through `config/azr_config.yaml`. Key parameters include:

- `model_id`: The base model to use (default: "Qwen/Qwen1.5-4B")
- `quantization`: Settings for 4-bit quantization
- `learning_rate`: Learning rate for model updates
- `max_steps`: Maximum number of training steps
- `task_types`: Types of tasks to generate (deduction, abduction, induction)
- `evaluation`: Benchmark settings

See the full configuration file for all available options.

## Usage

### Training

To start training the AZR model:

```bash
cd scripts
python train.py --config ../config/azr_config.yaml
```

To resume training from a checkpoint:

```bash
python train.py --config ../config/azr_config.yaml --resume ../models/checkpoint-1000
```

### Evaluation

To evaluate the model on coding benchmarks:

```bash
python evaluation.py --model_path ../models/checkpoint-1000 --benchmark humaneval
```

Available benchmarks:
- `humaneval`: OpenAI's HumanEval benchmark
- `mbpp`: Google's Mostly Basic Python Problems
- `apps`: APPS coding competition problems

### Inference

To use the trained model for solving coding problems:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/azr-checkpoint-final", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./models/azr-checkpoint-final")

prompt = "Write a Python function that returns the factorial of a number."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Process

The AZR training process follows these steps:

1. **Task Generation**: The system generates a programming task
2. **Task Solving**: The model attempts to solve the task
3. **Validation**: The solution is executed and validated
4. **Reinforcement**: The model is updated based on the success or failure
5. **Iteration**: The process repeats with increasingly difficult tasks

Training progress is tracked through Weights & Biases and local logs.

## Benchmarking

The system evaluates model performance on standard coding benchmarks:

- **HumanEval**: Tests code correctness from natural language descriptions
- **MBPP**: Tests Python programming from input-output examples
- **APPS**: Tests problem-solving and planning on competition problems

Benchmark results are saved to `logs/evaluation_results.json`.

## Directory Structure

```
azr/
├── data/                  # Task history and solutions
├── logs/                  # Training and evaluation logs
├── models/                # Model checkpoints
├── scripts/
│   ├── executor.py        # Safe code execution
│   ├── proposer.py        # Task generation
│   ├── solver.py          # Task solving
│   ├── train.py           # Training loop
│   ├── evaluation.py      # Benchmark evaluation
│   └── utils.py           # Utility functions
└── config/
    └── azr_config.yaml    # Configuration file
```

## Performance Optimization

For optimal performance on MacBook Pro M3 Max:

1. **Quantization**: The model uses 4-bit quantization to reduce memory usage
2. **Batch Size**: Default batch size is set to 2, adjust based on available memory
3. **Device Mapping**: The model automatically maps to available devices

## License

[MIT License](LICENSE)

## Citation

If you use AZR in your research, please cite:

```
@software{hernandez2025azr,
  author = {Hernandez, Joel},
  title = {Absolute Zero Reasoner: Self-Improving Code Generation},
  year = {2025},
  url = {https://github.com/joelhjames21/azr}
}
```

## Acknowledgements

- The Qwen team for the base model
- The Hugging Face team for the Transformers library
- The TRL team for reinforcement learning tools
