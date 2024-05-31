from typing import Dict, Tuple, Callable
import flwr as fl
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
import torch.optim as optim
from feddebug.utils.utils import get_data_loaders, trainFLMain
from torch.utils.data import DataLoader

class FedDebugClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader, testloader):
        print('Creating FedDebugClient')
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self,config = None) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        print('Fitting')
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        return float(loss / len(self.testloader)), len(self.testloader), {"accuracy": correct / len(self.testloader.dataset)}

