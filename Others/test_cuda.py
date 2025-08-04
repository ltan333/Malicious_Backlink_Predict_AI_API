import torch
print(torch.version.cuda)   # → None if CPU-only
print(torch.cuda.is_available())  # → False if do not have CUDA support