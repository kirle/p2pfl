#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Federated Averaging (FedAvg) Aggregator."""

from typing import List
from enum import Enum
import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.callbacks.requirements import CallbackRequirement


class AggregationMode(Enum):
    SIMPLE = 1
    CONTROL_VARIATE = 2
  
class ScaffoldAggregator(Aggregator):
    def __init__(self, node_name: str, total_nodes: int, eta_g: float):
        super().__init__(node_name)
        self.global_model = None  # Should be set initially
        self.global_control_variate = []  # List of tensors matching model parameters
        self.eta_g = eta_g  # Global learning rate
        self.total_nodes = total_nodes  # Total number of nodes
    
    @property
    def required_callbacks(self) -> List[CallbackRequirement]:
        return [CallbackRequirement.GRADIENT_CAPTURE]

    def set_initial_model(self, model: P2PFLModel):
        self.global_model = model
        self.global_control_variate = [torch.zeros_like(param) for param in model.get_parameters()]
    
    def aggregate(self, updates: List[Dict[str, Any]]) -> P2PFLModel:
        if not updates:
            raise NoModelsToAggregateError("No updates to aggregate.")
    
        # Aggregate delta_y_i
        delta_y_accum = [torch.zeros_like(param) for param in self.global_model.get_parameters()]
        for update in updates:
            delta_y_i = update['delta_y_i']
            for accum, delta_y in zip(delta_y_accum, delta_y_i):
                accum += delta_y  # Simple sum; consider weighted average if needed
    
        # Update global model
        new_global_weights = [
            param + self.eta_g * delta_y
            for param, delta_y in zip(self.global_model.get_parameters(), delta_y_accum)
        ]
        self.global_model.set_parameters(new_global_weights)
    
        # Aggregate delta_c_i
        delta_c_accum = sum([delta_c for update in updates for delta_c in update['delta_c_i']])
        scaling_factor = len(updates) / self.total_nodes
        # Update global control variate
        self.global_control_variate = [
            c + scaling_factor * delta_c_accum
            for c in self.global_control_variate
        ]
    
        return self.global_model
  
  
# class Scaffold(Aggregator):
#     """
#     Federated Averaging (FedAvg) [McMahan et al., 2016].

#     Paper: https://arxiv.org/abs/1602.05629.
#     """

#     def __init__(self, node_name: str):
#         super().__init__(node_name)
#         self.server_control_variate = None  # server control variate

#     @property
#     def required_callbacks(self) -> List[CallbackRequirement]:
#         return [CallbackRequirement.GRADIENT_CAPTURE]

#     def aggregate(self, models: List[P2PFLModel], aggregation_mode = AggregationMode.SIMPLE ) -> P2PFLModel:
#         if not models: 
#             raise NoModelsToAggregateError(f"({self.node_name}) Trying to aggregate models when there is no models")
        
#         if self.server_control_variate is None:
#             # using first model layers as reference
#             first_model_gradients = models[0].get_info('gradients')
#             if not first_model_gradients:
#                 raise ValueError(f"({self.node_name}) Trying to aggregate models when there is no gradients")
#             self.server_control_variate = [np.zeros_like(layer) for layer in first_model_gradients]
        
#         total_samples = sum(m.get_num_samples() for m in models) # total samples used in the aggregation
        
#         # init accumulators
#         first_model_weights = models[0].get_parameters()
#         aggregated_weights = [np.zeros_like(layer) for layer in first_model_weights]
#         aggregated_gradients = [np.zeros_like(layer) for layer in models[0].get_info('gradients')]
        
#         for model in models:
#             num_samples = model.get_num_samples() # samples used on this model
#             client_weights = model.get_parameters()
#             client_gradients = model.get_info('gradients')
            
#             # aggregate params weighted by the number of samples
#             for i, layer in enumerate(client_weights):
#                 aggregated_weights[i] += layer * num_samples
                
#             for i, layer in enumerate(client_gradients):
#                 aggregated_gradients[i] += layer * num_samples
            
#         averaged_weights = [layer / total_samples for layer in aggregated_weights]
#         averaged_gradients = [layer / total_samples for layer in aggregated_gradients]
        
#         # update server control variate
#         if aggregation_mode == AggregationMode.SIMPLE:
#             for i in range(len(self.server_control_variate)):
#                 self.server_control_variate[i] += averaged_gradients[i]
                
#         elif aggregation_mode == AggregationMode.CONTROL_VARIATE:
#             pass # TODO
            
#         # Get contributors
#         contributors = []
#         for model in models:
#             contributors.extend(model.get_contributors())

#         # Return an aggregated p2pfl model
#         aggregated_model = models[0].build_copy(
#             params=averaged_weights,
#             num_samples=total_samples,
#             contributors=contributors
#         )

#         return aggregated_model