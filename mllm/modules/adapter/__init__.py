from .q_former import AL_Adapter
from .linear import LinearAdapter
from .mlp import MLPAdapter

def load_adapter(adapter_type, adapter_config):
    if adapter_type == "linear":
        return LinearAdapter(**adapter_config)
    elif adapter_type == "mlp":
        return MLPAdapter(**adapter_config)
    elif adapter_type == "q_former":
        return AL_Adapter(**adapter_config)
    else:
        raise ValueError(f"Invalid adapter type: {adapter_type}")
