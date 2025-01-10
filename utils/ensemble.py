import torch

import utils.model_related as mr
from .log_writer import getLogWritter

logger = getLogWritter(__name__)

class Ensemble(torch.nn.Module):

    # dict_models: 
    #   -> clave: nombre del modelo
    #   -> valor: el modelo
    # weight_votations: clave: 
    #   -> clave: nombre del modelo
    #   -> valor: diccionario de clases tal que:
    #       -> la clave es el nombre/identificador de la clase
    #       -> el valor es el peso de la votación de ese modelo en esa clase
    # def __init__(self, dict_models: dict[str, torch.nn.Module], weight_votations: dict[str, dict[int, float]] = None, is_main_device:bool=False, *args, **kwargs):
    def __init__(self, dict_models:dict, weight_votations:dict = None, is_main_device:bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if is_main_device:
            logger.debug("Creating ensemble")
        self.is_main_device = is_main_device
        self.dict_models = dict_models
        self.weight_votations = weight_votations
        #for _, model in self.dict_models:
        #    # capas_entrenar_final=  1: Se entrena al menos el clasificador. 0, no se entrena nada. -1 se entrena todo.
        #    mr.transfer_learning(model=model, capas_entrenar_final = 1, is_main_device=is_main_device)

        if self.is_main_device:
            logger.debug(f"Ensemble created with the following models:\n{dict_models.keys()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        models_outputs = list()
        for name_model, model in self.dict_models:
            #models_outputs.append(output * self.weight_votations[name_model] )
            output = model(x)
            if self.is_main_device:
                logger.info(f"output:\n{models_outputs}")
                logger.info(f"type(output): {type(output)}")
            
            output *= self.weight_votations[name_model] 
            models_outputs.append(output)

        if self.is_main_device:
            logger.info(f"Models outputs - post weighted:\n{models_outputs}")
            logger.info(f"type(models_outputs[{name_model}]): {type(models_outputs[name_model])}")
        
        # Votations
        votations = models_outputs[0]
        for i in range(1, models_outputs):
            votations += self.weight_votations[i]
            
        # TODO: Revisar esta función 
        # Faltaría: 
        # - ver qué salida quiere en el caso de un clasificador y dársela.
        # - hacer un max de la clase más votada si espera 0 en una 1 en el resto, o, si no, normalizarla.

        if True:
            #Caso normalizar:
            result = (votations - min(votations)) / (max(votations) - min(votations))
        else:
            #Caso solo sacar el máximo
            result = max(votations)


        if self.is_main_device:
            logger.info(f"Votations results:\n{result}")        
            logger.info(f"max(votations):\n{max(votations)}")


        return result
    
    # TODO: hacer esto bien!
    # TODO: Posible solución, hacer que los parámetros del ensemble sean los de todos sus modelos en el constructor (ir sacándolos y añadiéndolos)
    # TODO: Alternativa. Poner un optimizador para cada modelo y aplicar cada optimizador a cada modelo (?)
    # Para que el optimizador no se queje de que no hay parámetros
    def parameters(self, recurse = True):

        #list_tensors_parameters = [torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()]
        
        print(f"self.dict_models.values():\n{self.dict_models.values()}")
        #list_tensors_parameters = [torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()]
        list_tensors_parameters = [torch.Tensor(module.parameters(recurse)) for module in self.dict_models.values()]

        print(f"list_tensors_parameters:\n{list_tensors_parameters}")

        return torch.concat(list_tensors_parameters)
        #return torch.concat([torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()])
        #return super().parameters(recurse)
    
