"""
Adapter modules for projecting audio features to LLM space.
"""
from .q_former import AL_Adapter
from .linear import LinearAdapter
from .mlp import MLPAdapter

def load_adapter(adapter_type: str, adapter_config: dict):
    """
    Load adapter module by type.

    Args:
        adapter_type: Type of adapter ('linear', 'mlp', or 'q_former')
        adapter_config: Configuration dictionary for adapter

    Returns:
        Initialized adapter module
    """
    if adapter_type == "linear":
        return LinearAdapter(**adapter_config)
    elif adapter_type == "mlp":
        return MLPAdapter(**adapter_config)
    elif adapter_type == "q_former":
        return AL_Adapter(**adapter_config)
    else:
        raise ValueError(f"Invalid adapter type: {adapter_type}")
