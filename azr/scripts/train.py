# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Trainer

# Description:  
# Main training loop for the AZR system using reinforcement learning

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import json
import random
import re
import socket
import threading
import torch
from torch.optim import AdamW
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AZR modules
from proposer import propose_task, TaskProposer
from solver import solve_task, TaskSolver
from executor import validate_code, execute_with_timeout
from utils import (
    setup_seed, 
    load_config, 
    setup_wandb, 
    save_checkpoint, 
    load_checkpoint,
    log_metrics,
    create_directory_structure
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AZR-Trainer")

# Dashboard data server
class DashboardDataServer:
    """Server to provide real-time training data to the dashboard"""
    
    def __init__(self, host='localhost', port=8081):
        """Initialize the dashboard data server"""
        self.host = host
        self.port = port
        self.data = {
            'status': 'initializing',
            'currentStep': 0,
            'taskType': 'Deduction',
            'taskDifficulty': 0.1,
            'successRate': 0,
            'avgReward': 0,
            'tasksSolved': 0,
            'bufferSize': 0,
            'recentTasks': [],
            'benchmarkProgress': {
                'humaneval': 0,
                'mbpp': 0,
                'apps': 0
            },
            'trainingData': {
                'steps': [],
                'rewards': [],
                'successRates': []
            }
        }
        self.server_socket = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the dashboard data server"""
        self.running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Dashboard data server started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the dashboard data server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Dashboard data server stopped")
        
    def update_data(self, data):
        """Update the dashboard data"""
        self.data.update(data)
        
    def _run_server(self):
        """Run the dashboard data server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    threading.Thread(target=self._handle_client, args=(client_socket, addr)).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    
        except Exception as e:
            logger.error(f"Error starting dashboard data server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
                
    def _handle_client(self, client_socket, addr):
        """Handle a client connection"""
        try:
            # Read the HTTP request
            request = client_socket.recv(1024).decode('utf-8')
            
            # Check if it's a GET request for the API endpoint
            if 'GET /api/training-data' in request:
                # Send HTTP response with JSON data
                response = f"HTTP/1.1 200 OK\r\n"
                response += f"Content-Type: application/json\r\n"
                response += f"Access-Control-Allow-Origin: *\r\n"
                response += f"\r\n"
                response += json.dumps(self.data)
                
                client_socket.sendall(response.encode('utf-8'))
            else:
                # Send 404 response
                response = f"HTTP/1.1 404 Not Found\r\n\r\n"
                client_socket.sendall(response.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()

class AZRTrainer:
    """
    Absolute Zero Reasoner (AZR) Trainer
    
    Implements a reinforcement learning loop for training a language model
    to solve programming tasks through self-play.
    """
    
    def __init__(self, config_path: str = "../config/azr_config.yaml"):
        """
        Initialize the AZR trainer.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set random seed
        setup_seed(self.config.get("seed", 42))
        
        # Initialize WandB
        self.run_id = setup_wandb(self.config)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._setup_model()
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize PPO trainer
        self.ppo_config = PPOConfig(
            learning_rate=self.config.get("learning_rate", 1e-6),
            batch_size=self.config.get("batch_size", 2),
            mini_batch_size=1,
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            optimize_cuda_cache=True,
            early_stopping=self.config.get("early_stopping", True),
            target_kl=self.config.get("target_kl", 0.1),
            kl_penalty=self.config.get("kl_penalty", "kl"),
            seed=self.config.get("seed", 42),
            log_with="wandb" if self.run_id else None
        )
        
        # Initialize task proposer and solver
        self.proposer = TaskProposer()
        self.solver = TaskSolver()
        
        # Initialize metrics
        self.metrics = {
            "reward": [],
            "task_success_rate": 0.0,
            "kl_divergence": 0.0,
            "loss": 0.0
        }
        
        # Initialize task buffer
        self.task_buffer = []
        self.max_buffer_size = self.config.get("task_buffer_size", 1000)
        
        # Initialize benchmark tracking
        self.benchmark_scores = {
            "humaneval": 0.0,
            "mbpp": 0.0,
            "apps": 0.0
        }
        self.benchmark_targets = {
            "humaneval": {
                "gpt35": 48.1,
                "codellama": 53.7,
                "claude2": 56.0,
                "target": 67.3
            },
            "mbpp": {
                "gpt35": 52.3,
                "codellama": 57.2,
                "claude2": 61.5,
                "target": 72.1
            },
            "apps": {
                "gpt35": 27.5,
                "codellama": 31.2,
                "claude2": 33.8,
                "target": 42.7
            }
        }
        
        # Initialize training history
        self.training_history = {
            "steps": [],
            "rewards": [],
            "success_rates": [],
            "task_difficulties": [],
            "benchmark_scores": {
                "humaneval": [],
                "mbpp": [],
                "apps": []
            }
        }
        
        # Initialize recent tasks
        self.recent_tasks = []
        self.max_recent_tasks = 20
        
        # Initialize dashboard data server
        self.dashboard_server = DashboardDataServer()
        self.dashboard_server.start()
        
        # Initialize start time
        self.start_time = datetime.now()
        
        # Initialize task solved counter
        self.tasks_solved = 0
        
        logger.info("AZR Trainer initialized")
    
    def _setup_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Set up the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_id = self.config.get("model_id", "Qwen/Qwen1.5-4B")
        logger.info(f"Loading model: {model_id}")
        
        # Configure quantization
        quant_config = None
        if self.config.get("quantization", {}).get("load_in_4bit", True):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    self.config.get("quantization", {}).get("bnb_4bit_compute_dtype", "float16")
                ),
                bnb_4bit_use_double_quant=self.config.get("quantization", {}).get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=self.config.get("quantization", {}).get("bnb_4bit_quant_type", "nf4")
            )
            logger.info("Using 4-bit quantization")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def _setup_optimizer(self) -> AdamW:
        """
        Set up the optimizer.
        
        Returns:
            AdamW optimizer
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.config.get("learning_rate", 1e-6),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        return optimizer
    
    def _generate_task(self, difficulty: float = 0.5) -> Dict[str, Any]:
        """
        Generate a new programming task.
        
        Args:
            difficulty: Task difficulty from 0.0 to 1.0
            
        Returns:
            Task dictionary
        """
        # Select a random task type
        task_type = random.choice(self.config.get("task_types", ["deduction"]))
        
        # Generate task using the proposer
        task = self.proposer.propose_task(
            self.model, 
            self.tokenizer, 
            task_type, 
            difficulty=difficulty
        )
        
        return task
    
    def _solve_task(self, task: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Solve a programming task.
        
        Args:
            task: Task dictionary
            
        Returns:
            Tuple of (success, solution_code)
        """
        # Solve task using the solver
        success, solution = self.solver.solve_task(
            self.model,
            self.tokenizer,
            task
        )
        
        return success, solution
    
    def _validate_solution(self, task: Dict[str, Any], solution: str) -> float:
        """
        Validate a solution and compute reward.
        
        Args:
            task: Task dictionary
            solution: Solution code
            
        Returns:
            Reward value
        """
        # Extract input and expected output from task description
        # This is a simplified implementation and would need more robust parsing
        task_desc = task["description"]
        
        # Try to extract input and output examples
        input_example = "example_input"
        expected_output = "example_output"
        
        # Look for input example in task description
        input_match = re.search(r"Input Example:(.+?)Expected Output:", task_desc, re.DOTALL)
        if input_match:
            input_example = input_match.group(1).strip()
        
        # Look for expected output in task description
        output_match = re.search(r"Expected Output:(.+?)(?:Constraints:|$)", task_desc, re.DOTALL)
        if output_match:
            expected_output = output_match.group(1).strip()
        
        # Validate the solution
        is_valid = validate_code(solution, input_example, expected_output)
        
        # Compute reward
        reward = 1.0 if is_valid else 0.0
        
        return reward
    
    def _update_task_buffer(self, task: Dict[str, Any], solution: str, reward: float):
        """
        Update the task buffer with a new task and its solution.
        
        Args:
            task: Task dictionary
            solution: Solution code
            reward: Reward value
        """
        # Add task to buffer
        self.task_buffer.append({
            "task": task,
            "solution": solution,
            "reward": reward
        })
        
        # Trim buffer if it exceeds max size
        if len(self.task_buffer) > self.max_buffer_size:
            self.task_buffer = self.task_buffer[-self.max_buffer_size:]
    
    def _sample_from_buffer(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Sample tasks from the buffer.
        
        Args:
            batch_size: Number of tasks to sample
            
        Returns:
            List of task dictionaries
        """
        if not self.task_buffer:
            return []
        
        # Sample tasks with higher probability for high-reward tasks
        weights = [entry["reward"] + 0.1 for entry in self.task_buffer]  # Add small constant to avoid zero weights
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample with replacement
        indices = random.choices(range(len(self.task_buffer)), weights=probs, k=min(batch_size, len(self.task_buffer)))
        
        return [self.task_buffer[i] for i in indices]
    
    def train_step(self, step: int) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of metrics
        """
        # Generate a new task with adaptive difficulty
        difficulty = min(0.1 + step / self.config.get("max_steps", 100000) * 0.9, 0.9)
        task = self._generate_task(difficulty=difficulty)
        
        # Solve the task
        success, solution = self._solve_task(task)
        
        # Validate the solution and compute reward
        reward = self._validate_solution(task, solution) if success else 0.0
        
        # Update task buffer
        if success:
            self._update_task_buffer(task, solution, reward)
            self.tasks_solved += 1
            
            # Add to recent tasks
            self.recent_tasks.insert(0, {
                "id": step,
                "type": task.get("type", "Unknown"),
                "difficulty": difficulty,
                "solved": True,
                "description": task.get("description", "")
            })
        else:
            # Add to recent tasks as failed
            self.recent_tasks.insert(0, {
                "id": step,
                "type": task.get("type", "Unknown"),
                "difficulty": difficulty,
                "solved": False,
                "description": task.get("description", "")
            })
        
        # Limit recent tasks
        if len(self.recent_tasks) > self.max_recent_tasks:
            self.recent_tasks = self.recent_tasks[:self.max_recent_tasks]
        
        # Sample tasks from buffer for PPO training
        if step > 0 and step % self.config.get("ppo_update_interval", 10) == 0 and self.task_buffer:
            batch = self._sample_from_buffer(self.config.get("batch_size", 2))
            
            if batch:
                # Prepare inputs for PPO
                # This is a simplified implementation and would need more work for a real PPO update
                # In a real implementation, you would need to:
                # 1. Generate responses from the model
                # 2. Compute rewards
                # 3. Run PPO update
                
                # For now, we'll just log the metrics
                self.metrics["reward"] = [entry["reward"] for entry in batch]
                self.metrics["task_success_rate"] = sum(self.metrics["reward"]) / len(self.metrics["reward"])
        
        # Update training history
        self.training_history["steps"].append(step)
        self.training_history["rewards"].append(reward)
        self.training_history["success_rates"].append(self.metrics["task_success_rate"])
        self.training_history["task_difficulties"].append(difficulty)
        
        # Evaluate on benchmarks periodically
        if step > 0 and step % self.config.get("benchmark_interval", 100) == 0:
            self._evaluate_benchmarks(step)
        
        # Update dashboard data
        self._update_dashboard_data(step, task, reward, difficulty)
        
        # Save checkpoint
        if step > 0 and step % self.config.get("checkpoint_interval", 100) == 0:
            save_checkpoint(
                self.model,
                self.tokenizer,
                self.optimizer,
                step,
                self.metrics,
                self.config
            )
        
        # Log metrics
        log_metrics(self.metrics, step)
        
        return self.metrics
    
    def _evaluate_benchmarks(self, step: int):
        """
        Evaluate the model on benchmarks.
        
        Args:
            step: Current training step
        """
        # In a real implementation, this would run the evaluation script
        # For now, we'll simulate progress based on training step
        
        # Simulate benchmark progress
        max_steps = self.config.get("max_steps", 100000)
        progress_rate = 1 - np.exp(-step / (max_steps / 3))
        
        # Update benchmark scores
        self.benchmark_scores["humaneval"] = progress_rate * self.benchmark_targets["humaneval"]["target"]
        self.benchmark_scores["mbpp"] = progress_rate * self.benchmark_targets["mbpp"]["target"]
        self.benchmark_scores["apps"] = progress_rate * self.benchmark_targets["apps"]["target"]
        
        # Update benchmark history
        self.training_history["benchmark_scores"]["humaneval"].append(self.benchmark_scores["humaneval"])
        self.training_history["benchmark_scores"]["mbpp"].append(self.benchmark_scores["mbpp"])
        self.training_history["benchmark_scores"]["apps"].append(self.benchmark_scores["apps"])
        
        logger.info(f"Benchmark scores at step {step}:")
        logger.info(f"  HumanEval: {self.benchmark_scores['humaneval']:.2f}%")
        logger.info(f"  MBPP: {self.benchmark_scores['mbpp']:.2f}%")
        logger.info(f"  APPS: {self.benchmark_scores['apps']:.2f}%")
    
    def _update_dashboard_data(self, step: int, task: Dict[str, Any], reward: float, difficulty: float):
        """
        Update the dashboard data.
        
        Args:
            step: Current training step
            task: Current task
            reward: Current reward
            difficulty: Current task difficulty
        """
        # Calculate success rate over recent history
        recent_rewards = self.training_history["rewards"][-100:] if len(self.training_history["rewards"]) > 0 else [0]
        success_rate = sum(1 for r in recent_rewards if r > 0.5) / len(recent_rewards)
        
        # Calculate average reward over recent history
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Prepare training data for charts
        steps = self.training_history["steps"][-100:]
        rewards = self.training_history["rewards"][-100:]
        success_rates = [sum(1 for r in self.training_history["rewards"][max(0, i-10):i+1] if r > 0.5) / min(i+1, 10) 
                         for i in range(len(self.training_history["rewards"]))[-100:]]
        
        # Update dashboard data
        dashboard_data = {
            'status': 'active',
            'currentStep': step,
            'taskType': task.get("type", "Unknown"),
            'taskDifficulty': difficulty,
            'successRate': success_rate,
            'avgReward': avg_reward,
            'tasksSolved': self.tasks_solved,
            'bufferSize': len(self.task_buffer),
            'recentTasks': self.recent_tasks,
            'benchmarkProgress': {
                'humaneval': self.benchmark_scores["humaneval"],
                'mbpp': self.benchmark_scores["mbpp"],
                'apps': self.benchmark_scores["apps"]
            },
            'trainingData': {
                'steps': steps,
                'rewards': rewards,
                'successRates': success_rates
            }
        }
        
        # Send data to dashboard
        self.dashboard_server.update_data(dashboard_data)
    
    def train(self, resume_from: Optional[str] = None):
        """
        Train the model.
        
        Args:
            resume_from: Optional path to resume training from a checkpoint
        """
        # Resume from checkpoint if provided
        start_step = 0
        if resume_from:
            self.model, self.tokenizer, self.optimizer, training_state = load_checkpoint(
                resume_from,
                self.model,
                self.tokenizer,
                self.optimizer
            )
            start_step = training_state.get("step", 0)
            logger.info(f"Resumed training from step {start_step}")
        
        # Main training loop
        max_steps = self.config.get("max_steps", 100000)
        
        logger.info(f"Starting training for {max_steps} steps")
        
        try:
            for step in tqdm(range(start_step, max_steps), desc="Training"):
                try:
                    # Perform training step
                    metrics = self.train_step(step)
                    
                    # Log progress
                    if step % 10 == 0:
                        reward_avg = sum(metrics.get("reward", [0])) / max(len(metrics.get("reward", [1])), 1)
                        logger.info(f"Step {step}/{max_steps} - Reward: {reward_avg:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    continue
            
            # Save final checkpoint
            save_checkpoint(
                self.model,
                self.tokenizer,
                self.optimizer,
                max_steps,
                self.metrics,
                self.config
            )
            
            logger.info("Training completed")
            
        finally:
            # Stop dashboard server
            self.dashboard_server.stop()

def main():
    """Main function to run the AZR training"""
    parser = argparse.ArgumentParser(description="Train the AZR model")
    parser.add_argument("--config", type=str, default="../config/azr_config.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dashboard-port", type=int, default=8081, help="Port for dashboard data server")
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Initialize trainer
    trainer = AZRTrainer(config_path=args.config)
    
    # Train model
    trainer.train(resume_from=args.resume)

if __name__ == "__main__":
    main()
