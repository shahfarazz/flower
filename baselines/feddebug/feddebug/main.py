import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import flwr as fl

# from feddebug.client import client_fn as create_client_fn
from feddebug.strategy import FedDebugStrategy
from feddebug.utils.utils import get_data_loaders
from feddebug.client import FedDebugClient

from feddebug.models import Net

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    trainloader, testloader = get_data_loaders(cfg.dataset.name, cfg.dataset.batch_size)

    def get_client(pid: int) -> fl.client.Client:
        #TODO make trainloader testloader separate for each client 
        net = Net()
        return FedDebugClient(net,trainloader, testloader)


   

    # 4. Define your strategy
    strategy = FedDebugStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
    )

    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,  # Sample 100% of available clients for training
    #     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    #     min_fit_clients=10,  # Never sample less than 10 clients for training
    #     min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    #     min_available_clients=10,  # Wait until all 10 clients are available
    # )

    # fl.server.stra

    # 5. Start Simulation
    fl.simulation.start_simulation(
        client_fn= get_client,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 8},
        client_resources={"num_cpus": 1},
    )

if __name__ == "__main__":
    main()