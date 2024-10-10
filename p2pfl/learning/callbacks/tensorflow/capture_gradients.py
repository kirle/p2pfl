from typing import Dict
import tensorflow as tf
from lightning.pytorch.callbacks import Callback
from p2pfl.learning.callbacks.factory import CallbackFactory
from p2pfl.learning.callbacks.requirements import CallbackRequirement

@CallbackFactory.register_callback(requirement=CallbackRequirement.GRADIENT_CAPTURE)
class CaptureGradientsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.gradients: Dict[str, tf.Tensor] = {}
    
    def on_train_epoch_end(self, trainer, pl_module):
        for layer in pl_module.model.layers:
            for weight in layer.weights:
                if hasattr(weight, 'gradient') and weight.gradient is not None:
                    self.gradients[weight.name] = weight.gradient.numpy().copy()
    
    def get_gradients(self) -> Dict[str, tf.Tensor]:
        return self.gradients