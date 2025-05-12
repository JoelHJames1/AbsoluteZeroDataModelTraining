#!/usr/bin/env python3
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: AZRTrainer

# Description:  
# Main training loop for the Absolute Zero Reasoner (AZR) system

import os
import sys
import time
import json
import random
import argparse
import socket
import threading
import logging
import yaml
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.proposer import propose_task
from scripts.solver import solve_task
from scripts.executor import validate_code

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AZR-Trainer")

class AZRTrainer:
    """Main trainer class for the Absolute Zero Reasoner system"""
    
    def __init__(self, config_path=None):
        """Initialize the trainer with the given configuration"""
        # Load configuration
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "azr_config.yaml"
        )
        self.load_config()
        
        # Create necessary directories
        self.setup_directories()
        
        # Initialize training state
        self.initialize_state()
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize task buffer
        self.task_buffer = []
        
        # Start data server
        self.start_data_server()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def setup_directories(self):
        """Create necessary directories for logs, data, and models"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create directories if they don't exist
        for dir_name in ["logs", "data", "models"]:
            dir_path = os.path.join(base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory {dir_path} is ready")
    
    def initialize_state(self):
        """Initialize the training state"""
        self.state = {
            'current_step': 0,
            'task_type': self.config['task_types'][0],
            'task_difficulty': self.config.get('initial_difficulty', 0.1),
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'tasks_solved': 0,
            'buffer_size': 0,
            'recent_tasks': [],
            'benchmark_progress': {
                'humaneval': 0.0,
                'mbpp': 0.0,
                'apps': 0.0
            },
            'training_data': {
                'steps': [],
                'rewards': [],
                'successRates': []
            }
        }
        logger.info("Training state initialized")
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model {self.config['model_id']}...")
            
            # Configure quantization
            quant_config = None
            if 'quantization' in self.config:
                quant_params = self.config['quantization']
                # Skip quantization if explicitly disabled
                if 'enabled' in quant_params and not quant_params['enabled']:
                    logger.info("Quantization is disabled in config")
                    quant_config = None
                # 8-bit quantization
                elif 'load_in_8bit' in quant_params and quant_params['load_in_8bit']:
                    try:
                        quant_config = BitsAndBytesConfig(
                            load_in_8bit=True
                        )
                        logger.info("Using 8-bit quantization")
                    except ImportError:
                        logger.warning("bitsandbytes not available, disabling quantization")
                        quant_config = None
                # 4-bit quantization
                elif 'load_in_4bit' in quant_params and quant_params['load_in_4bit']:
                    try:
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=getattr(torch, quant_params.get('bnb_4bit_compute_dtype', 'float16')),
                            bnb_4bit_use_double_quant=quant_params.get('bnb_4bit_use_double_quant', True),
                            bnb_4bit_quant_type=quant_params.get('bnb_4bit_quant_type', 'nf4')
                        )
                        logger.info("Using 4-bit quantization")
                    except ImportError:
                        logger.warning("bitsandbytes not available, disabling quantization")
                        quant_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_id'],
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_id'],
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Configure LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                # Target modules for Qwen3 model
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, peft_config)
            
            # Configure PPO
            ppo_config = PPOConfig(
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                mini_batch_size=1,
                gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
                # Removed optimize_cuda_cache as it's not supported in this version
                early_stopping=self.config.get('early_stopping', True),
                target_kl=self.config.get('target_kl', 0.1),
                kl_penalty=self.config.get('kl_penalty', "kl"),
                seed=self.config.get('seed', 42)
            )
            
            # Initialize PPO trainer
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def start_data_server(self):
        """Start a server to provide training data to the dashboard"""
        self.server_thread = threading.Thread(target=self._run_data_server, daemon=True)
        self.server_thread.start()
        logger.info("Data server started")
    
    def _run_data_server(self):
        """Run the data server to provide training data"""
        host = 'localhost'
        port = 8081
        
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((host, port))
            server_socket.listen(5)
            logger.info(f"Data server listening on {host}:{port}")
            
            while True:
                client_socket, addr = server_socket.accept()
                logger.info(f"Connection from {addr}")
                
                # Handle client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()
        except Exception as e:
            logger.error(f"Error in data server: {e}")
        finally:
            server_socket.close()
    
    def _handle_client(self, client_socket):
        """Handle client connection to the data server"""
        try:
            # Receive request
            request = client_socket.recv(1024).decode()
            
            # Check if it's a request for training data
            if "GET /api/training-data" in request:
                # Prepare response
                response_body = json.dumps(self.state)
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    f"Content-Length: {len(response_body)}\r\n"
                    "\r\n"
                    f"{response_body}"
                )
                
                # Send response
                client_socket.sendall(response.encode())
            else:
                # Send 404 for other requests
                response = "HTTP/1.1 404 Not Found\r\n\r\n"
                client_socket.sendall(response.encode())
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def update_task_difficulty(self):
        """Update task difficulty based on success rate"""
        if self.state['success_rate'] > 0.8:
            # Increase difficulty if success rate is high
            new_difficulty = min(
                self.state['task_difficulty'] + 0.05,
                self.config.get('final_difficulty', 0.9)
            )
            self.state['task_difficulty'] = new_difficulty
            logger.info(f"Increased task difficulty to {new_difficulty:.2f}")
        elif self.state['success_rate'] < 0.3:
            # Decrease difficulty if success rate is low
            new_difficulty = max(
                self.state['task_difficulty'] - 0.02,
                self.config.get('initial_difficulty', 0.1)
            )
            self.state['task_difficulty'] = new_difficulty
            logger.info(f"Decreased task difficulty to {new_difficulty:.2f}")
    
    def select_task_type(self):
        """Select a task type from the configured types"""
        # Occasionally change task type
        if random.random() < 0.1:
            task_type = random.choice(self.config['task_types'])
            self.state['task_type'] = task_type
            logger.info(f"Selected task type: {task_type}")
        return self.state['task_type']
    
    def generate_task(self):
        """Generate a new task"""
        task_type = self.select_task_type()
        difficulty = self.state['task_difficulty']
        
        # Generate task using the proposer
        task_prompt = f"Generate a {task_type} Python programming task with difficulty {difficulty:.2f} on a scale of 0.1 to 0.9."
        task_description = propose_task(self.model, self.tokenizer, task_prompt)
        
        # Create task object
        task = {
            'id': self.state['current_step'],
            'type': task_type,
            'difficulty': difficulty,
            'description': task_description,
            'solved': False
        }
        
        logger.info(f"Generated task: {task['description'][:50]}...")
        return task
    
    def solve_task(self, task):
        """Attempt to solve the given task"""
        # Prepare prompt for the solver
        solve_prompt = f"Task: {task['description']}\n\nWrite a Python function to solve this task:"
        
        # Generate solution
        solution = solve_task(self.model, self.tokenizer, solve_prompt)
        
        # Extract code from solution
        code = self._extract_code(solution)
        
        # Validate solution
        is_valid = validate_code(code, task['description'], "")
        
        # Update task with solution and result
        task['solution'] = solution
        task['code'] = code
        task['solved'] = is_valid
        
        logger.info(f"Task {task['id']} solved: {is_valid}")
        return task, is_valid
    
    def _extract_code(self, solution):
        """Extract code from a solution text"""
        # Look for code blocks
        if "```python" in solution and "```" in solution.split("```python", 1)[1]:
            code = solution.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in solution and "```" in solution.split("```", 1)[1]:
            code = solution.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            # Just use the whole solution if no code blocks found
            code = solution
        
        # Ensure code has a function definition
        if not code.strip().startswith("def "):
            code = "def f(input_data):\n    " + "\n    ".join(code.split("\n"))
        
        return code
    
    def update_training_data(self, reward):
        """Update training data with the latest step results"""
        self.state['current_step'] += 1
        
        # Update success metrics
        if len(self.state['training_data']['rewards']) > 0:
            # Calculate moving average
            avg_reward = (
                0.9 * self.state['avg_reward'] + 
                0.1 * reward
            )
            self.state['avg_reward'] = avg_reward
        else:
            self.state['avg_reward'] = reward
        
        # Update success rate
        if reward > 0:
            self.state['tasks_solved'] += 1
        
        self.state['success_rate'] = (
            self.state['tasks_solved'] / 
            max(1, self.state['current_step'])
        )
        
        # Add data point for charts (every 5 steps to avoid too many points)
        if self.state['current_step'] % 5 == 0:
            self.state['training_data']['steps'].append(self.state['current_step'])
            self.state['training_data']['rewards'].append(self.state['avg_reward'])
            self.state['training_data']['successRates'].append(self.state['success_rate'])
            
            # Limit data points to keep dashboard responsive
            max_points = 50
            if len(self.state['training_data']['steps']) > max_points:
                self.state['training_data']['steps'] = self.state['training_data']['steps'][-max_points:]
                self.state['training_data']['rewards'] = self.state['training_data']['rewards'][-max_points:]
                self.state['training_data']['successRates'] = self.state['training_data']['successRates'][-max_points:]
        
        # Update buffer size
        self.state['buffer_size'] = len(self.task_buffer)
        
        # Log progress
        logger.info(
            f"Step {self.state['current_step']}: "
            f"Reward={reward:.2f}, "
            f"Success Rate={self.state['success_rate']:.2f}, "
            f"Avg Reward={self.state['avg_reward']:.2f}"
        )
    
    def update_benchmark_progress(self):
        """Simulate benchmark progress based on training progress"""
        # In a real implementation, this would run actual benchmarks
        # For now, we'll simulate progress based on success rate
        
        # Only update occasionally to simulate benchmark evaluation intervals
        if self.state['current_step'] % self.config.get('evaluation_interval', 500) != 0:
            return
        
        # Simulate benchmark progress
        progress_factor = min(1.0, self.state['current_step'] / 10000)
        success_factor = self.state['success_rate']
        
        # Calculate progress for each benchmark
        humaneval_target = self.config['benchmark_targets']['humaneval']['target']
        mbpp_target = self.config['benchmark_targets']['mbpp']['target']
        apps_target = self.config['benchmark_targets']['apps']['target']
        
        # Update benchmark progress
        self.state['benchmark_progress'] = {
            'humaneval': progress_factor * success_factor * humaneval_target,
            'mbpp': progress_factor * success_factor * mbpp_target,
            'apps': progress_factor * success_factor * apps_target * 0.7  # APPS is harder
        }
        
        logger.info(
            f"Benchmark progress: "
            f"HumanEval={self.state['benchmark_progress']['humaneval']:.2f}%, "
            f"MBPP={self.state['benchmark_progress']['mbpp']:.2f}%, "
            f"APPS={self.state['benchmark_progress']['apps']:.2f}%"
        )
    
    def save_checkpoint(self):
        """Save a model checkpoint"""
        if self.state['current_step'] % self.config.get('checkpoint_interval', 100) != 0:
            return
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            f"checkpoint-{self.state['current_step']}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.info(f"Checkpoint saved at step {self.state['current_step']}")
    
    def ppo_update(self, task, reward):
        """Update the model using PPO based on the task and reward"""
        if self.state['current_step'] % self.config.get('ppo_update_interval', 10) != 0:
            return
        
        try:
            # Prepare inputs for PPO update
            prompt = f"Task: {task['description']}\n\nWrite a Python function to solve this task:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response with the model
            response_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            ).sequences
            
            # Decode response
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # Prepare query and response for PPO
            query_tensor = inputs.input_ids
            response_tensor = response_ids[:, inputs.input_ids.shape[1]:]
            
            # Create rewards tensor
            rewards = torch.tensor([reward]).to(self.model.device)
            
            # Perform PPO step
            stats = self.ppo_trainer.step(query_tensor, response_tensor, rewards)
            
            logger.info(f"PPO update at step {self.state['current_step']}: {stats}")
        except Exception as e:
            logger.error(f"Error in PPO update: {e}")
    
    def train(self, max_steps=None):
        """Main training loop"""
        max_steps = max_steps or self.config.get('max_steps', 100000)
        
        logger.info(f"Starting training for {max_steps} steps")
        
        try:
            while self.state['current_step'] < max_steps:
                # Generate a new task
                task = self.generate_task()
                
                # Attempt to solve the task
                task, solved = self.solve_task(task)
                
                # Calculate reward
                reward = 1.0 if solved else -0.1
                
                # Update model with PPO
                self.ppo_update(task, reward)
                
                # Add task to buffer and recent tasks
                self.task_buffer.append(task)
                if len(self.task_buffer) > self.config.get('task_buffer_size', 1000):
                    self.task_buffer.pop(0)
                
                # Update recent tasks list
                self.state['recent_tasks'].insert(0, task)
                if len(self.state['recent_tasks']) > 20:
                    self.state['recent_tasks'].pop()
                
                # Update training data
                self.update_training_data(reward)
                
                # Update task difficulty
                self.update_task_difficulty()
                
                # Update benchmark progress
                self.update_benchmark_progress()
                
                # Save checkpoint
                self.save_checkpoint()
                
                # Sleep briefly to avoid overwhelming the system
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint()
            logger.info("Training completed")

def main():
    """Main function to run the AZR trainer"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AZR Trainer')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--steps', type=int, help='Number of training steps')
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = AZRTrainer(config_path=args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        # TODO: Implement checkpoint loading
        pass
    
    # Start training
    trainer.train(max_steps=args.steps)

if __name__ == "__main__":
    main()
