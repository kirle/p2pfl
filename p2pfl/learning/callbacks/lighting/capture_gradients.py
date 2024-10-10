from typing import Dict
import torch
from lightning.pytorch.callbacks import Callback
from p2pfl.learning.callbacks.factory import CallbackFactory
from p2pfl.learning.callbacks.requirements import CallbackRequirement

@CallbackFactory.register_callback(requirement=CallbackRequirement.GRADIENT_CAPTURE)
class CaptureGradientsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.gradients: Dict[str, torch.Tensor] = {}
        
    def on_train_epoch_end(self, pl_module):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                self.gradients[name] = param.grad.clone().detach() 
        
    def get_gradients(self):
        return self.gradients
        