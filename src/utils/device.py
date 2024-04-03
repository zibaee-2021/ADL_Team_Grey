import torch

def get_optimal_device():
    """
    Returns: optimal device.
        CUDA (GPU) if available, MPS if mac, CPU
    """
    if torch.cuda.is_available():
        ret = "cuda"
    elif torch.backends.mps.is_available():
        ret = "mps"
    else:
        ret = "cpu"
    print(f"Running on {ret}")
    return ret