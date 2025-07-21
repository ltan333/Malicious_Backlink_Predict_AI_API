import torch
print(torch.version.cuda)   # → None nếu là bản CPU-only
print(torch.cuda.is_available())  # → False nếu không có CUDA support