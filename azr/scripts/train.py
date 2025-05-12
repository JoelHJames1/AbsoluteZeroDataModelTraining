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
import torch
from torch.optim import AdamW
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

def main():
    """Main function to run the AZR training"""
    parser = argparse.ArgumentParser(description="Train the AZR model")
    parser.add_argument("--config", type=str, default="../config/azr_config.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Initialize trainer
    trainer = AZRTrainer(config_path=args.config)
    
    # Train model
    trainer.train(resume_from=args.resume)

if __name__ == "__main__":
    main()
