# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Solver

# Description:  
# Solves programming tasks using the AZR model

import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AZR-Solver")

class TaskSolver:
    """
    Solves programming tasks using the AZR model.
    Implements various prompting strategies to improve solution quality.
    """
    
    def __init__(self, solutions_path: str = "../data/solutions.json"):
        """
        Initialize the task solver.
        
        Args:
            solutions_path: Path to the JSON file containing previous solutions
        """
        self.solutions_path = solutions_path
        self.solutions = self._load_solutions()
        
        # Prompting strategies
        self.strategies = [
            self._direct_solution_prompt,
            self._step_by_step_prompt,
            self._test_driven_prompt,
            self._chain_of_thought_prompt
        ]
    
    def _load_solutions(self) -> Dict[int, Dict[str, Any]]:
        """Load solutions from JSON file if it exists"""
        if os.path.exists(self.solutions_path):
            try:
                with open(self.solutions_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading solutions: {e}")
                return {}
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.solutions_path), exist_ok=True)
            return {}
    
    def _save_solutions(self):
        """Save solutions to JSON file"""
        try:
            with open(self.solutions_path, 'w') as f:
                json.dump(self.solutions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving solutions: {e}")
    
    def _direct_solution_prompt(self, task: Dict[str, Any]) -> str:
        """Generate a direct solution prompt"""
        return f"""
Solve the following Python programming task:

{task['description']}

Write a Python function named 'f' that solves this problem.
Your solution should be efficient, well-commented, and handle edge cases.

def f(
"""
    
    def _step_by_step_prompt(self, task: Dict[str, Any]) -> str:
        """Generate a step-by-step solution prompt"""
        return f"""
Solve the following Python programming task step by step:

{task['description']}

1. First, understand the problem:
   - What are the inputs and outputs?
   - What are the constraints?
   - What edge cases should be considered?

2. Plan your approach:
   - What algorithm or data structure is appropriate?
   - What is the time and space complexity?

3. Implement the solution as a function named 'f':

def f(
"""
    
    def _test_driven_prompt(self, task: Dict[str, Any]) -> str:
        """Generate a test-driven solution prompt"""
        return f"""
Solve the following Python programming task using test-driven development:

{task['description']}

First, write test cases based on the examples and edge cases:
```python
# Test cases
assert f(...) == ...
assert f(...) == ...
```

Now implement the function 'f' that passes all test cases:

def f(
"""
    
    def _chain_of_thought_prompt(self, task: Dict[str, Any]) -> str:
        """Generate a chain-of-thought solution prompt"""
        return f"""
Solve the following Python programming task by reasoning through each step:

{task['description']}

Let's think through this problem:
1. What is the problem asking for?
2. What are some examples and their expected outputs?
3. What patterns can we identify?
4. How can we translate these patterns into code?

Based on this reasoning, here's the implementation:

def f(
"""
    
    def _extract_function(self, solution_text: str) -> Optional[str]:
        """Extract the function definition from the solution text"""
        # Look for a Python function definition
        match = re.search(r'def\s+f\s*\(.*?(?:return|pass).*?(?:\n\S|\Z)', 
                         solution_text, re.DOTALL)
        
        if match:
            return match.group(0)
        
        # If no match, look for code blocks
        code_blocks = re.findall(r'```python(.*?)```', solution_text, re.DOTALL)
        for block in code_blocks:
            if 'def f(' in block:
                return block.strip()
        
        # Last resort: look for any function definition
        match = re.search(r'def\s+\w+\s*\(.*?(?:return|pass).*?(?:\n\S|\Z)', 
                         solution_text, re.DOTALL)
        if match:
            # Replace the function name with 'f'
            func_def = match.group(0)
            func_name = re.search(r'def\s+(\w+)', func_def).group(1)
            return func_def.replace(f'def {func_name}', 'def f')
        
        return None
    
    def solve_task(self, model, tokenizer, task: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Solve a programming task using the model.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            task: Task dictionary containing description and other details
            
        Returns:
            Tuple of (success, solution_code)
        """
        # Try different prompting strategies
        for strategy_fn in self.strategies:
            prompt = strategy_fn(task)
            
            # Generate solution using the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.2,  # Lower temperature for more focused solutions
                top_p=0.95
            )
            solution_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the function from the solution text
            solution_code = self._extract_function(solution_text)
            
            if solution_code:
                # Store the solution
                if isinstance(task.get('id'), int):
                    self.solutions[str(task['id'])] = {
                        "task_id": task['id'],
                        "solution": solution_code,
                        "strategy": strategy_fn.__name__,
                        "full_response": solution_text
                    }
                    self._save_solutions()
                
                return True, solution_code
        
        # If all strategies fail
        logger.warning(f"Failed to generate a valid solution for task {task.get('id', 'unknown')}")
        return False, ""

def solve_task(model, tokenizer, task_prompt: str) -> str:
    """
    Simple function wrapper around TaskSolver for compatibility with the original code.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        task_prompt: Task description text
        
    Returns:
        Solution code as text
    """
    # Create a task dictionary from the prompt
    task = {"description": task_prompt, "id": "temp"}
    
    solver = TaskSolver()
    success, solution = solver.solve_task(model, tokenizer, task)
    
    return solution if success else "# Failed to generate a solution"

if __name__ == "__main__":
    # Example usage (without actual model)
    print("Task Solver module - run with a model to solve tasks")
