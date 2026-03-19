import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is actually available to PyTorch
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # Quick test: Move a small tensor to the GPU
    x = torch.rand(5, 3).to("cuda")
    print("\nSuccess! A tensor was moved to the RTX 4050.")
    print(x)
else:
    print("\nCUDA is NOT available. PyTorch is running on the CPU.")
    print("Tip: You might have the 'CPU-only' version of Torch installed.")