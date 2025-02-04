import torch


def init_gpu():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set to the first GPU
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
