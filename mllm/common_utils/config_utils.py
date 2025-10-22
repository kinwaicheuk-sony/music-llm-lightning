"""
Configuration and checkpoint utilities.
"""
import json
import torch
from safetensors.torch import load_file
import importlib

def load_ckpt_state_dict(ckpt_path: str) -> dict:
    """
    Load checkpoint state dictionary from file.

    Args:
        ckpt_path: Path to checkpoint file (.safetensors or .pth)

    Returns:
        State dictionary
    """
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    return state_dict

def instantiate_from_config(config: dict, **kwargs):
    """
    Instantiate object from configuration dictionary.

    Args:
        config: Configuration dict with 'target' and 'params'
        **kwargs: Additional keyword arguments

    Returns:
        Instantiated object
    """
    return get_obj_from_str(config["target"])(
        **config.get("params", dict()),
        **kwargs
    )

def get_obj_from_str(string: str):
    """
    Import and return object from module string.

    Args:
        string: Module path string (e.g., 'torch.nn.Linear')

    Returns:
        Class or function object
    """
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    return state_dict


def create_optimizer_from_config(optimizer_config: dict, parameters) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        optimizer_config: Optimizer configuration dictionary
        parameters: Parameters to optimize

    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_config["type"]
    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config: dict, optimizer: torch.optim.Optimizer):
    """
    Create learning rate scheduler from config.

    Args:
        scheduler_config: Scheduler configuration dictionary
        optimizer: Optimizer instance

    Returns:
        Configured scheduler
    """
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler
