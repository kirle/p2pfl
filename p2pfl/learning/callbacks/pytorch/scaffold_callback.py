import copy
from typing import Dict
import torch
import lightning as L
from p2pfl.learning.aggregators.scaffold  import Scaffold
from p2pfl.learning.callbacks.factory import CallbackFactory
from p2pfl.learning.callbacks.requirements import CallbackRequirement
from p2pfl.learning.p2pfl_model import P2PFLModel

@CallbackFactory.register_callback(requirement=CallbackRequirement.SCAFFOLD)
class SCAFFOLDCallback(L.pytorch.callbacks.Callback):
    def __init__(self, aggregator: Scaffold):
        super().__init__()
        self.c_i = None
        self.aggregator = aggregator
        self.c = None
        self.global_control_variate_updated = False
        self.saved_lr = None
        self.K = 0 # nº local steps
        
    def on_train_start(self, pl_module: P2PFLModel):
        if self.c_i is None:
            self.c_i = [torch.zeros_like(param) for param in pl_module.parameters()]
        if not self.global_control_variate_updated:
            self._get_global_c(pl_module)
            
        self.initial_model_weights = copy.deepcopy(pl_module.model.get_parameters())
        
    
    def on_before_optimizer_step(self, pl_module: P2PFLModel):
        # save current learning rate because it can be changed by the optimizer
        self.saved_lr = pl_module.optimizer.param_groups[0]['lr']
     
    def on_after_optimizer_step(self, pl_module: P2PFLModel):
        c = self.c
        eta_l = self.saved_lr
        
        # Apply control variate adjustment: y_i ← y_i + eta_l * c_i - eta_l * c
        # Equivalent to y_i ← y_i - eta_l * (-c_i + c)
        for param, c_i_param, c_param in zip(pl_module.parameters(), self.c_i, c):
            if param.grad is not None:
                param.data += eta_l * c_i_param
                param.data -= eta_l * c_param
                
        self.K += 1
        
    def on_train_end(self, pl_module: P2PFLModel):
        y_i = pl_module.model.get_parameters()
        x_g = self.initial_model_weights
        
        previous_c_i = [c.clone() for c in self.c_i] # save previous c_i for delta_c_i
        
        # update local control variate: c_i <- c_i - (x_g - y_i) / (K * eta_l)
        self.c_i = [
            c_i - (x - y) / (self.K * self.saved_lr)
            for c_i, x, y in zip(self.c_i, x_g, y_i)
        ]
        
        # delta_y_i = y_i - x_g
        delta_y_i = [y - x for y, x in zip(y_i, x_g)]
        
        # delta_c_i = c_i - c
        delta_c_i = [c_new - c_old for c_new, c_old in zip(self.c_i, previous_c_i)]
        
        pl_module.model.additional_info['delta_y_i'] = delta_y_i
        pl_module.model.additional_info['delta_c_i'] = delta_c_i
    
    def _get_global_c(self, pl_module: P2PFLModel):
        # get c from aggregator
        self.c = [
            torch.from_numpy(c_np).to(pl_module.device) 
            for c_np in self.aggregator.global_control_variate
        ]
        self.global_control_variate_updated = True
    
        