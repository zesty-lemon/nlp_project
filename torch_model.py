import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the number of available CUDA devices
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")

    # Get the name of the first CUDA device
    device_name = torch.cuda.get_device_name(0)
    print(f"Device name: {device_name}")
