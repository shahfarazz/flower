"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def prepare_dataset(name: str, batch_size: int):
    """Prepare the dataset based on the given name and batch size."""

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=2)

        return trainloader, testloader

    else:
        raise ValueError(f"Dataset {name} is not supported.")

# You can add more datasets following the same pattern if needed
