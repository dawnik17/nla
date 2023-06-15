import torch


def get_torch_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
