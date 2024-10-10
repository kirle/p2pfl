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

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.callbacks.requirements import CallbackRequirement


class Scaffold(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016].

    Paper: https://arxiv.org/abs/1602.05629.
    """
    @property
    def required_callbacks(self) -> List[CallbackRequirement]:
        return [CallbackRequirement.GRADIENT_CAPTURE]
    
    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        """
        pass
