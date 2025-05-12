# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Proposer

# Description:  
# Generates new programming tasks for the AZR model to solve

import random
import json
import os
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AZR-Proposer")

class TaskProposer:
    """
    Generates programming tasks of varying difficulty for the AZR model to solve.
    Uses a combination of templates and previous successful tasks to create new challenges.
    """
    
    def __init__(self, task_history_path: str = "../data/task_history.json"):
        """
        Initialize the task proposer.
        
        Args:
            task_history_path: Path to the JSON file containing task history
        """
        self.task_history_path = task_history_path
        self.task_history = self._load_task_history()
        
        # Task templates by type and difficulty
        self.task_templates = {
            "deduction": [
                "Write a Python function that {action} {subject}.",
                "Implement a function to {action} given {subject}.",
                "Create a Python solution that {action} {subject} efficiently."
            ],
            "abduction": [
                "Given {observation}, write a function that explains {subject}.",
                "Based on {observation}, implement a function that determines {subject}.",
                "Create a function that identifies the cause of {observation} related to {subject}."
            ],
            "induction": [
                "Write a function that generalizes the pattern: {examples} for any {subject}.",
                "Implement a solution that extends the following pattern: {examples}.",
                "Create a function that predicts the next items in the sequence: {examples}."
            ]
        }
        
        # Task components for template filling
        self.components = {
            "action": [
                "calculates", "sorts", "finds", "optimizes", "transforms", 
                "validates", "generates", "analyzes", "converts", "compresses"
            ],
            "subject": [
                "an array of integers", "a binary tree", "a string", "a matrix", 
                "a graph", "a linked list", "a set of points", "a sequence", 
                "a dictionary", "numeric data"
            ],
            "observation": [
                "a series of events", "input-output pairs", "system behavior", 
                "performance metrics", "error patterns", "user interactions"
            ],
            "examples": [
                "1,3,6,10,15", "a→b, b→c, c→?", "f(1)=1, f(2)=4, f(3)=9", 
                "[1,2], [2,3], [3,5], [?]", "triangle(1)=1, triangle(2)=3, triangle(3)=6"
            ]
        }
    
    def _load_task_history(self) -> List[Dict[str, Any]]:
        """Load task history from JSON file if it exists"""
        if os.path.exists(self.task_history_path):
            try:
                with open(self.task_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading task history: {e}")
                return []
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.task_history_path), exist_ok=True)
            return []
    
    def _save_task_history(self):
        """Save task history to JSON file"""
        try:
            with open(self.task_history_path, 'w') as f:
                json.dump(self.task_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving task history: {e}")
    
    def _fill_template(self, template: str, task_type: str) -> str:
        """Fill a template with random components"""
        result = template
        
        # Replace each placeholder with a random component
        for component, options in self.components.items():
            if "{" + component + "}" in result:
                result = result.replace("{" + component + "}", random.choice(options))
                
        return result
    
    def propose_task(self, model, tokenizer, task_type: str, difficulty: float = 0.5, 
                    references: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a new programming task.
        
        Args:
            model: The language model to use for task generation
            tokenizer: The tokenizer for the language model
            task_type: Type of task (deduction, abduction, induction)
            difficulty: Task difficulty from 0.0 to 1.0
            references: Optional list of reference tasks to build upon
            
        Returns:
            Dict containing task details
        """
        # Validate task type
        if task_type not in self.task_templates:
            task_type = random.choice(list(self.task_templates.keys()))
        
        # Select a template based on task type
        template = random.choice(self.task_templates[task_type])
        
        # Fill the template with components
        task_description = self._fill_template(template, task_type)
        
        # Use the model to enhance the task description
        prompt = f"""
Generate a detailed Python programming task based on this description:
"{task_description}"

The task should:
1. Be clear and specific
2. Include example input and expected output
3. Have a difficulty level of {difficulty*10}/10
4. Require algorithmic thinking
5. Be solvable with a single Python function named 'f'

FORMAT:
Task: [task title]
Description: [detailed description]
Input Example: [example input]
Expected Output: [expected output]
Constraints: [any constraints or requirements]
"""
        
        # Generate the enhanced task using the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        enhanced_task = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the task components (this is simplified and would need more robust parsing)
        task_parts = enhanced_task.split("\n")
        task_data = {
            "id": len(self.task_history) + 1,
            "type": task_type,
            "difficulty": difficulty,
            "description": enhanced_task,
            "created_at": "2025-05-11",  # In a real implementation, use datetime.now()
            "solved": False,
            "solution": None
        }
        
        # Add to task history
        self.task_history.append(task_data)
        self._save_task_history()
        
        return task_data

def propose_task(model, tokenizer, task_type, references=None):
    """
    Simple function wrapper around TaskProposer for compatibility with the original code.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        task_type: Type of task
        references: Optional reference tasks
        
    Returns:
        Task description text
    """
    proposer = TaskProposer()
    task = proposer.propose_task(model, tokenizer, task_type, references=references)
    return task["description"]

if __name__ == "__main__":
    # Example usage (without actual model)
    print("Task Proposer module - run with a model to generate tasks")
