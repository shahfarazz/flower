import flwr as fl
from flwr.common import FitRes, NDArrays
from typing import List, Tuple
import numpy as np

from feddebug.utils.faulty_client_localization.FaultyClientLocalization import FaultyClientLocalization
from feddebug.utils.faulty_client_localization.InferenceGuidedInputs import InferenceGuidedInputs

class FedDebugStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #cant be none
        self.generated_inputs = []
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> NDArrays:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        client_models = {client.cid: res.parameters for client, res in results}

        # Fault localization logic
        potential_faulty_clients, input_gen_time, fault_localization_time = FaultyClientLocalization.runFaultLocalization(
            self= self,
            na_t= 0.003,  # Provide appropriate threshold
            num_bugs=1  # Specify the number of bugs if known
        )

        print(f"Potential Faulty Clients: {potential_faulty_clients}")
        return aggregated_weights
