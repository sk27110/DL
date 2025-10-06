import torch
from torchvision import transforms

def denormalize(tensor, mean, std):
    """
    Преобразует нормализованный тензор обратно в обычный диапазон [0,1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean
