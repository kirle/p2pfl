from typing import Dict, List, Tuple, Type, Any

import logging
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.learner import NodeLearner
from lightning.pytorch.callbacks import Callback
from p2pfl.learning.callbacks.requirements import CallbackRequirement

class CallbackFactory:
    
    _registry: Dict[Tuple[CallbackRequirement, str], Type[Any]] = {} # mapping (requirement, framework) -> callback

    
    @classmethod
    def register_callback(cls, requirement:CallbackRequirement):
        
        def decorator(callback_cls:Type[Any]):
            framework = callback_cls.__module__.lower()
            key = (requirement, framework)    
            
            if key in cls._registry:
                logging.error(
                    f"Callback for requirement '{requirement.name}' and framework '{framework}' is already registered as "
                    f"'{cls._registry[key].__name__}'. Cannot register '{callback_cls.__name__}'."
                )
                raise ValueError(f"Callback {callback_cls} is already registered for requirement {requirement} and framework {framework}")
            
            cls._registry[key] = callback_cls
            return callback_cls
        
        return decorator
    
    @classmethod
    def create_callbacks (cls, learner:NodeLearner, aggr: Aggregator) -> List[Any]:
        framework = learner.__class__.__module__.lower()
        required_callbacks = aggr.required_callbacks
        callbacks = []
        
        for requirement in required_callbacks:
            key = (requirement, framework)
            if key in cls._registry:
                callback_cls = cls._registry[key]
                callbacks.append(callback_cls())
            else:
                logging.warning(
                    f"No callback registered for requirement '{requirement.name}' and framework '{framework}'."
                )
        return callbacks