import psutil
import platform
import os

def get_system_info():
    """Returns a dictionary containing hardware system information."""
    info = {
        "cpu_cores": psutil.cpu_count(logical=True) or 1,
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": False,
        "gpu_memory_gb": 0.0
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            # Get memory of the first GPU
            mem_info = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = mem_info / (1024**3)
    except Exception:
        # Ignore any torch import or CUDA errors safely
        pass

    return info

def get_device():
    """Safely determines the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def scale_value(score, min_val, max_val):
    """Linearly interpolate between min_val and max_val based on score (0 to 1)."""
    return int(min_val + score * (max_val - min_val))

def get_adaptive_nas_config(system_info):
    """
    Returns optimized NAS parameters based on hardware score.
    Score is calculated deterministically without randomness.
    """
    cores = system_info.get("cpu_cores", 1)
    ram = system_info.get("ram_gb", 1.0)
    gpu_mem = system_info.get("gpu_memory_gb", 0.0)
    has_gpu = system_info.get("gpu_available", False)

    # 1. Compute Hardware Score (0.0 to 1.0)
    cpu_score = min(1.0, cores / 16.0)
    ram_score = min(1.0, ram / 32.0)

    if has_gpu:
        gpu_score = min(1.0, gpu_mem / 24.0)
        score = 0.3 * cpu_score + 0.3 * ram_score + 0.4 * gpu_score
        score = min(1.0, score)

        # GPU ranges
        epochs = scale_value(score, 30, 60)
        population = scale_value(score, 30, 60)
        generations = scale_value(score, 15, 30)

        # Batch size based on GPU memory safely
        # E.g. 4GB -> 32, 24GB -> 256
        batch_size = 32
        if gpu_mem >= 24:
            batch_size = 256
        elif gpu_mem >= 16:
            batch_size = 128
        elif gpu_mem >= 8:
            batch_size = 64

    else:
        score = 0.5 * cpu_score + 0.5 * ram_score
        score = min(1.0, score)

        # CPU ranges (High-end vs Low-end defined by score/capacity)
        if cores >= 8 and ram >= 16:
            epochs = scale_value(score, 15, 30)
            population = scale_value(score, 15, 30)
            generations = scale_value(score, 8, 15)
            batch_size = 32
        else:
            epochs = scale_value(score, 5, 15)
            population = scale_value(score, 5, 15)
            generations = scale_value(score, 3, 8)
            batch_size = 16

    return {
        "epochs": clamp(epochs, 1, 1000),
        "population": clamp(population, 2, 200),
        "generations": clamp(generations, 1, 100),
        "batch_size": clamp(batch_size, 1, 512),
        "hardware_score": score
    }
