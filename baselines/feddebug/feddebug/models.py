from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):
    """ Base class for any CNN model. """

    def __init__(self, num_classes: int = 10):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Net(BaseModel):
    """ Simple CNN model for image classification. """

    def __init__(self, num_classes: int = 10):
        super(Net, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
