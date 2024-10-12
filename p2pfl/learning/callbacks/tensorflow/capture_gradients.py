from typing import Dict
import tensorflow as tf

from p2pfl.learning.callbacks.factory import CallbackFactory
from p2pfl.learning.callbacks.requirements import CallbackRequirement
from p2pfl.learning.p2pfl_model import P2PFLModel

@CallbackFactory.register_callback(requirement=CallbackRequirement.GRADIENT_CAPTURE)
class CaptureGradientsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_end(self, pl_module: P2PFLModel):
        for layer in pl_module.model.layers:
            for weight in layer.weights:
                if hasattr(weight, 'gradient') and weight.gradient is not None:
                    self.pl_module.model.additional_info['gradients'] = self.gradients
    
