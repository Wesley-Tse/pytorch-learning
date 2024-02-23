import torch

print("torch version:", torch.__version__)

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print("Number Of Available GPUs:", num_devices)
    for i in range(num_devices):
        device = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i} Detail:")
        print("Name:", device.name)
        print("Computing Power:", f"{device.major}.{device.minor}")
        print("Memory (GB):", round(device.total_memory / (1024**3), 1))
else:
    print("No Available GPU !")
