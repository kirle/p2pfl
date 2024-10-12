import copy
from typing import List, Dict, Any
import numpy as np
import torch
import lightning as L
from p2pfl.learning.aggregators.scaffold import ScaffoldAggregator
from p2pfl.learning.callbacks.factory import CallbackFactory
from p2pfl.learning.callbacks.requirements import CallbackRequirement
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError


class ScaffoldAggregator(Aggregator):
    def __init__(self, node_name: str, eta_g: float):
        super().__init__(node_name)
        self.global_model: P2PFLModel = None
        self.global_control_variate = []
        self.eta_g = eta_g

    @property
    def required_callbacks(self) -> List[CallbackRequirement]:
        return [CallbackRequirement.GRADIENT_CAPTURE, CallbackRequirement.SCAFFOLD]
    
    def get_global_control_variate(self): # property?
        return self.global_control_variate

    def set_initial_model(self, model: P2PFLModel):
        self.global_model = model
        self.global_control_variate = [np.zeros_like(param) for param in model.get_parameters()]

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        if not models:
            raise NoModelsToAggregateError("No models to aggregate.")
        

        delta_y_accum = [np.zeros_like(param) for param in self.global_model.get_parameters()]
        delta_c_accum = [np.zeros_like(c_variate) for c_variate in self.global_control_variate]

        for model in models:
            delta_y_i = model.get_info("delta_y_i") 
            delta_c_i = model.get_info("delta_c_i")
            
            for idx, (accum, delta_y) in enumerate(zip(delta_y_accum, delta_y_i)):
                if delta_y is not None and isinstance(delta_y, np.ndarray):
                    accum += delta_y
                else:
                    raise ValueError(f"Client delta_y_i at parameter index {idx} is invalid.")

            for idx, (accum, delta_c) in enumerate(zip(delta_c_accum, delta_c_i)):
                if delta_c is not None and isinstance(delta_c, np.ndarray):
                    accum += delta_c
                else:
                    raise ValueError(f"Client delta_c_i at parameter index {idx} is invalid.")
        
        # x ← x + eta_g * delta_y_i
        new_global_weights = [
            param + self.eta_g * delta_y
            for param, delta_y in zip(self.global_model.get_parameters(), delta_y_accum)
        ]
        self.global_model.set_parameters(new_global_weights)

        scaling_factor = 1 # TODO: Check if S=N 

        # global_c ← global_c + (S / N) * delta_c_i
        self.global_control_variate = [
            c + scaling_factor * delta_c
            for c, delta_c in zip(self.global_control_variate, delta_c_accum)
        ]

        return self.global_model