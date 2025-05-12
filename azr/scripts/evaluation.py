# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Evaluator

# Description:  
# Evaluates the AZR model on standard coding benchmarks

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AZR modules
from executor import validate_code, execute_with_timeout
from utils import setup_seed, load_config, log_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AZR-Evaluator")

class ModelEvaluator:
    """
    Evaluates the AZR model on standard coding benchmarks.
    Supports HumanEval, MBPP, and APPS benchmarks.
    """
    
    def __init__(self, model_path: str, config_path: str = "../config/azr_config.yaml"):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the model checkpoint
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set random seed
        setup_seed(self.config.get("seed", 42))
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(model_path)
        
        # Benchmark datasets
        self.benchmarks = {
            "humaneval": self._load_humaneval,
            "mbpp": self._load_mbpp,
            "apps": self._load_apps
        }
        
        # Metrics
        self.metrics = {}
        
        logger.info(f"Model evaluator initialized with model from {model_path}")
    
    def _load_model(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {model_path}")
        
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
        if os.path.exists(model_path):
            # Load from local path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Load from model ID
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        return model, tokenizer
    
    def _load_humaneval(self) -> List[Dict[str, Any]]:
        """
        Load the HumanEval benchmark dataset.
        
        Returns:
            List of task dictionaries
        """
        try:
            dataset = load_dataset("openai_humaneval")
            
            tasks = []
            for item in dataset["test"]:
                task = {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "entry_point": item["entry_point"],
                    "test": item["test"],
                    "canonical_solution": item["canonical_solution"]
                }
                tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} tasks from HumanEval")
            return tasks
        except Exception as e:
            logger.error(f"Error loading HumanEval: {e}")
            return []
    
    def _load_mbpp(self) -> List[Dict[str, Any]]:
        """
        Load the MBPP benchmark dataset.
        
        Returns:
            List of task dictionaries
        """
        try:
            dataset = load_dataset("mbpp")
            
            tasks = []
            for item in dataset["test"]:
                task = {
                    "task_id": f"mbpp_{item['task_id']}",
                    "prompt": item["text"],
                    "test_cases": item["test_list"],
                    "canonical_solution": item["code"]
                }
                tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} tasks from MBPP")
            return tasks
        except Exception as e:
            logger.error(f"Error loading MBPP: {e}")
            return []
    
    def _load_apps(self) -> List[Dict[str, Any]]:
        """
        Load the APPS benchmark dataset.
        
        Returns:
            List of task dictionaries
        """
        try:
            dataset = load_dataset("codeparrot/apps", split="test")
            
            tasks = []
            for i, item in enumerate(dataset):
                task = {
                    "task_id": f"apps_{i}",
                    "prompt": item["question"],
                    "difficulty": item["difficulty"],
                    "test_cases": item["input_output"]
                }
                tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} tasks from APPS")
            return tasks
        except Exception as e:
            logger.error(f"Error loading APPS: {e}")
            return []
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from model output.
        
        Args:
            text: Model output text
            
        Returns:
            Extracted code
        """
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for function definitions
        functions = re.findall(r'def\s+\w+\s*\(.*?(?:return|pass).*?(?:\n\S|\Z)', text, re.DOTALL)
        if functions:
            return '\n'.join(functions)
        
        # Return the whole text as a fallback
        return text
    
    def _evaluate_humaneval(self, tasks: List[Dict[str, Any]], 
                           num_samples: int = 1) -> Dict[str, float]:
        """
        Evaluate the model on HumanEval.
        
        Args:
            tasks: List of HumanEval tasks
            num_samples: Number of samples to generate per task
            
        Returns:
            Dictionary of metrics
        """
        correct = 0
        total = 0
        
        for task in tqdm(tasks, desc="Evaluating HumanEval"):
            prompt = task["prompt"]
            
            # Generate solutions
            solutions = []
            for _ in range(num_samples):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
                )
                solution_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract code from solution
                solution = self._extract_code(solution_text)
                solutions.append(solution)
            
            # Evaluate solutions
            for solution in solutions:
                # Combine prompt and solution
                full_code = prompt + solution
                
                # Create test code
                test_code = full_code + "\n\n" + task["test"]
                
                # Execute test code
                success, _, _ = execute_with_timeout(test_code)
                
                if success:
                    correct += 1
                
                total += 1
        
        # Calculate metrics
        pass_rate = correct / total if total > 0 else 0
        
        metrics = {
            "pass@1": pass_rate,
            "correct": correct,
            "total": total
        }
        
        return metrics
    
    def _evaluate_mbpp(self, tasks: List[Dict[str, Any]], 
                      num_samples: int = 1) -> Dict[str, float]:
        """
        Evaluate the model on MBPP.
        
        Args:
            tasks: List of MBPP tasks
            num_samples: Number of samples to generate per task
            
        Returns:
            Dictionary of metrics
        """
        correct = 0
        total = 0
        
        for task in tqdm(tasks, desc="Evaluating MBPP"):
            prompt = f"Write a Python function to solve the following problem:\n{task['prompt']}\n\ndef "
            
            # Generate solutions
            solutions = []
            for _ in range(num_samples):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
                )
                solution_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract code from solution
                solution = "def " + self._extract_code(solution_text)
                solutions.append(solution)
            
            # Evaluate solutions
            for solution in solutions:
                # Check if solution passes all test cases
                all_passed = True
                
                for test_case in task["test_cases"]:
                    # Create test code
                    test_code = solution + "\n\n" + test_case
                    
                    # Execute test code
                    success, _, _ = execute_with_timeout(test_code)
                    
                    if not success:
                        all_passed = False
                        break
                
                if all_passed:
                    correct += 1
                
                total += 1
        
        # Calculate metrics
        pass_rate = correct / total if total > 0 else 0
        
        metrics = {
            "pass@1": pass_rate,
            "correct": correct,
            "total": total
        }
        
        return metrics
    
    def _evaluate_apps(self, tasks: List[Dict[str, Any]], 
                      num_samples: int = 1) -> Dict[str, float]:
        """
        Evaluate the model on APPS.
        
        Args:
            tasks: List of APPS tasks
            num_samples: Number of samples to generate per task
            
        Returns:
            Dictionary of metrics
        """
        correct = 0
        total = 0
        
        # Group tasks by difficulty
        tasks_by_difficulty = {
            "introductory": [],
            "interview": [],
            "competition": []
        }
        
        for task in tasks:
            if task["difficulty"] == 0:
                tasks_by_difficulty["introductory"].append(task)
            elif task["difficulty"] == 1:
                tasks_by_difficulty["interview"].append(task)
            else:
                tasks_by_difficulty["competition"].append(task)
        
        # Metrics by difficulty
        metrics_by_difficulty = {}
        
        for difficulty, difficulty_tasks in tasks_by_difficulty.items():
            correct_difficulty = 0
            total_difficulty = 0
            
            for task in tqdm(difficulty_tasks, desc=f"Evaluating APPS ({difficulty})"):
                prompt = f"Solve the following programming problem:\n\n{task['prompt']}\n\n"
                
                # Generate solutions
                solutions = []
                for _ in range(num_samples):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=1024,  # APPS problems can be complex
                        temperature=0.2,
                        top_p=0.95
                    )
                    solution_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract code from solution
                    solution = self._extract_code(solution_text)
                    solutions.append(solution)
                
                # Evaluate solutions
                for solution in solutions:
                    # Check if solution passes all test cases
                    all_passed = True
                    
                    for test_case in task["test_cases"]:
                        input_data = test_case["input"]
                        expected_output = test_case["output"]
                        
                        # Create a wrapper to execute the solution with the input
                        wrapper_code = f"""
{solution}

# Execute with input
import sys
from io import StringIO

# Redirect stdout
original_stdout = sys.stdout
sys.stdout = StringIO()

# Prepare input
sys.stdin = StringIO('''{input_data}''')

# Run the solution
try:
    main()  # Assuming the solution has a main() function
except NameError:
    # If main() doesn't exist, try to execute the solution directly
    exec('''{solution}''')

# Get output
output = sys.stdout.getvalue()

# Restore stdout and stdin
sys.stdout = original_stdout
sys.stdin = sys.__stdin__

# Store result
result = output.strip()
"""
                        
                        # Execute wrapper code
                        success, _, local_vars = execute_with_timeout(wrapper_code, timeout=10)
                        
                        if not success or local_vars.get("result", "").strip() != expected_output.strip():
                            all_passed = False
                            break
                    
                    if all_passed:
                        correct += 1
                        correct_difficulty += 1
                    
                    total += 1
                    total_difficulty += 1
            
            # Calculate metrics for this difficulty
            pass_rate = correct_difficulty / total_difficulty if total_difficulty > 0 else 0
            
            metrics_by_difficulty[difficulty] = {
                "pass@1": pass_rate,
                "correct": correct_difficulty,
                "total": total_difficulty
            }
        
        # Calculate overall metrics
        pass_rate = correct / total if total > 0 else 0
        
        metrics = {
            "pass@1": pass_rate,
            "correct": correct,
            "total": total,
            "by_difficulty": metrics_by_difficulty
        }
        
        return metrics
    
    def evaluate(self, benchmark: str, num_samples: int = 1, 
                max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a benchmark.
        
        Args:
            benchmark: Benchmark name (humaneval, mbpp, apps)
            num_samples: Number of samples to generate per task
            max_tasks: Maximum number of tasks to evaluate
            
        Returns:
            Dictionary of metrics
        """
        # Check if benchmark is supported
        if benchmark not in self.benchmarks:
            logger.error(f"Benchmark {benchmark} not supported")
            return {}
        
        # Load benchmark tasks
        tasks = self.benchmarks[benchmark]()
        
        # Limit number of tasks if specified
        if max_tasks is not None and max_tasks > 0:
            tasks = tasks[:max_tasks]
        
        # Evaluate based on benchmark
        if benchmark == "humaneval":
            metrics = self._evaluate_humaneval(tasks, num_samples)
        elif benchmark == "mbpp":
            metrics = self._evaluate_mbpp(tasks, num_samples)
        elif benchmark == "apps":
            metrics = self._evaluate_apps(tasks, num_samples)
        else:
            metrics = {}
        
        # Store metrics
        self.metrics[benchmark] = metrics
        
        # Log metrics
        logger.info(f"Evaluation results for {benchmark}:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")
        
        return metrics
    
    def save_results(self, output_path: str = "../logs/evaluation_results.json"):
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate the AZR model on coding benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--benchmark", type=str, default="humaneval", 
                        choices=["humaneval", "mbpp", "apps"], help="Benchmark to evaluate on")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per task")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of tasks to evaluate")
    parser.add_argument("--output", type=str, default="../logs/evaluation_results.json", 
                        help="Path to save evaluation results")
    parser.add_argument("--config", type=str, default="../config/azr_config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, config_path=args.config)
    
    # Run evaluation
    evaluator.evaluate(args.benchmark, num_samples=args.num_samples, max_tasks=args.max_tasks)
    
    # Save results
    evaluator.save_results(args.output)

if __name__ == "__main__":
    main()
