# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Utils

# Description:  
# Utility functions for the AZR training system

import os
import json
import logging
import time
import torch
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import yaml
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/azr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AZR-Utils")

def setup_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def load_config(config_path: str = "../config/azr_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "learning_rate": 1e-6,
            "batch_size": 2,
            "task_types": ["deduction", "abduction", "induction"],
            "checkpoint_interval": 100,
            "max_steps": 100000,
            "evaluation_interval": 500,
            "wandb_project": "azr-training",
            "wandb_entity": None,
            "seed": 42,
            "model_id": "Qwen/Qwen1.5-4B",
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        }

def setup_wandb(config: Dict[str, Any]) -> str:
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WandB run ID
    """
    try:
        run = wandb.init(
            project=config.get("wandb_project", "azr-training"),
            entity=config.get("wandb_entity", None),
            config=config,
            name=f"azr-run-{time.strftime('%Y%m%d-%H%M%S')}",
            dir="../logs"
        )
        logger.info(f"WandB initialized with run ID: {run.id}")
        return run.id
    except Exception as e:
        logger.error(f"Error initializing WandB: {e}")
        return None

def save_checkpoint(model, tokenizer, optimizer, step: int, 
                   metrics: Dict[str, float], config: Dict[str, Any],
                   checkpoint_dir: str = "../models"):
    """
    Save model checkpoint and training state.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        optimizer: The optimizer state
        step: Current training step
        metrics: Dictionary of metrics
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
    """
    try:
        # Create checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        
        # Save training state
        training_state = {
            "step": step,
            "metrics": metrics,
            "config": config,
            "timestamp": time.time()
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved at step {step} to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return None

def load_checkpoint(checkpoint_path: str, model, tokenizer, optimizer=None):
    """
    Load model checkpoint and training state.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model: The model to load weights into
        tokenizer: The tokenizer to load
        optimizer: Optional optimizer to load state
        
    Returns:
        Tuple of (model, tokenizer, optimizer, training_state)
    """
    try:
        # Load model and tokenizer
        model = model.from_pretrained(checkpoint_path)
        tokenizer = tokenizer.from_pretrained(checkpoint_path)
        
        # Load optimizer state if provided
        if optimizer and os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
        
        # Load training state
        training_state = {}
        if os.path.exists(os.path.join(checkpoint_path, "training_state.json")):
            with open(os.path.join(checkpoint_path, "training_state.json"), 'r') as f:
                training_state = json.load(f)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return model, tokenizer, optimizer, training_state
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return model, tokenizer, optimizer, {}

def log_metrics(metrics: Dict[str, float], step: int, prefix: str = ""):
    """
    Log metrics to WandB and local log file.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        prefix: Optional prefix for metric names
    """
    try:
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to WandB if initialized
        if wandb.run is not None:
            wandb.log(metrics, step=step)
        
        # Log to local log file
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} - {metric_str}")
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")

def create_directory_structure():
    """Create the necessary directory structure for the AZR system"""
    try:
        directories = [
            "../data",
            "../logs",
            "../models",
            "../config"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create default config if it doesn't exist
        config_path = "../config/azr_config.yaml"
        if not os.path.exists(config_path):
            default_config = {
                "learning_rate": 1e-6,
                "batch_size": 2,
                "task_types": ["deduction", "abduction", "induction"],
                "checkpoint_interval": 100,
                "max_steps": 100000,
                "evaluation_interval": 500,
                "wandb_project": "azr-training",
                "wandb_entity": None,
                "seed": 42,
                "model_id": "Qwen/Qwen1.5-4B",
                "quantization": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            logger.info(f"Created default configuration at {config_path}")
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")

if __name__ == "__main__":
    # Example usage
    create_directory_structure()
    config = load_config()
    print(f"Loaded configuration: {config}")
