"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def prepare_datasets(dataset_name: str, batch_size: int, data_dir: str = "./data"):
    """create the Federated Dataset abstraction that from flwr-datasets that partitions the CIFAR-10."""
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)

        testset = CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=2)

        return trainloader, testloader
