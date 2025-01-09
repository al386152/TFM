import os

import torch

import utils.constants as cons
from .log_writer import getLogWritter

logger = getLogWritter(__name__)

# Plan: pasarle una lista de modelos y que hagan una votación de qué clase es una determinada entrada.

class Ensemble(torch.nn.Module):
    def __init__(self, list_models: list[torch.nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_models = list_models
        # TODO: Hacer esta función
        # Nota: en el "tutorial" (ejemplo que puso alguien en el foro de Torch), congelan os parámetros de los modelos: for param in model.parameters(): param.requires_grad_(False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Hacer esta función
        return x
    