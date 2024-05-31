import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dotmap import DotMap
from pytorch_lightning import seed_everything
from typing import Dict, List, Tuple

from feddebug.utils.faulty_client_localization.FaultyClientLocalization import FaultyClientLocalization


def get_data_loaders(name: str, batch_size: int):
    """Get data loaders for the given dataset name and batch size."""
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Added std values
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

def evaluateFaultLocalization(predicted_faulty_clients_on_each_input, true_faulty_clients):
    true_faulty_clients = set(true_faulty_clients)
    detection_acc = 0
    for pred_faulty_clients in predicted_faulty_clients_on_each_input:
        print(f"+++ Faulty Clients {pred_faulty_clients}")
        correct_localize_faults = len(
            true_faulty_clients.intersection(pred_faulty_clients))
        acc = (correct_localize_faults / len(true_faulty_clients)) * 100
        detection_acc += acc
    fault_localization_acc = detection_acc / len(predicted_faulty_clients_on_each_input)
    return fault_localization_acc

def runFaultyClientLocalization(client2models, exp2info, num_bugs, random_generator=torch.nn.init.kaiming_uniform_, apply_transform=True, k_gen_inputs=10, na_threshold=0.003, use_gpu=True):
    from utils.faulty_client_localization.InferenceGuidedInputs import InferenceGuidedInputs
    from utils.faulty_client_localization.FaultyClientLocalization import FaultyClientLocalization

    print(">  Running FaultyClientLocalization ..")
    input_shape = list(exp2info['data_config']['single_input_shape'])
    generate_inputs = InferenceGuidedInputs(client2models, input_shape, randomGenerator=random_generator, apply_transform=apply_transform,
                                            dname=exp2info['data_config']['name'], min_nclients_same_pred=5, k_gen_inputs=k_gen_inputs)
    selected_inputs, input_gen_time = generate_inputs.getInputs()

    start = time.time()
    faultyclientlocalization = FaultyClientLocalization(
        client2models, selected_inputs, use_gpu=use_gpu)

    potential_benign_clients_for_each_input = faultyclientlocalization.runFaultLocalization(
        na_threshold, num_bugs=num_bugs)
    fault_localization_time = time.time() - start
    return potential_benign_clients_for_each_input, input_gen_time, fault_localization_time

def trainFLMain(args: Dict):
    """Run federated learning training main function."""
    from utils.FLSimulation import trainFLMain as train

    # Prepare arguments as DotMap
    args = DotMap(args)

    # Log initial info
    print(f"FL Training with {args.num_clients} clients")

    # Run FL training
    c2ms, exp2info = train(args)
    client2models = {k: v.model.eval() for k, v in c2ms.items()}

    # Fault localization to find potential faulty clients
    #def runFaultLocalization(self, na_t, num_bugs):
    potential_faulty_clients, _, _ = FaultyClientLocalization.runFaultLocalization(
        self= None,
        na_t=0.003,
        num_bugs=1,)

    # Evaluate fault localization accuracy
    acc = evaluateFaultLocalization(
        potential_faulty_clients, exp2info['faulty_clients_ids'])
    print(f"Fault Localization Accuracy: {acc}")
