import torch


def device():
    """CUDA if available, else Apple-silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
