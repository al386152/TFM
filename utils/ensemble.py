import torch

import utils.model_related as mr
from .log_writer import getLogWritter

from collections import OrderedDict

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
    def __init__(self, dict_models:dict, weight_votations:dict = None, is_main_device:bool = False, num_classes = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if is_main_device:
            logger.debug("Creating ensemble")
        self.is_main_device = is_main_device
        self.dict_models = dict_models
        self.weight_votations = weight_votations
        self.num_classes = num_classes
        #for _, model in self.dict_models:
        #    # capas_entrenar_final=  1: Se entrena al menos el clasificador. 0, no se entrena nada. -1 se entrena todo.
        #    mr.transfer_learning(model=model, capas_entrenar_final = 1, is_main_device=is_main_device)

        if self.is_main_device:
            logger.debug(f"Ensemble created with the following models:\n{dict_models.keys()}")

    def get_list_models(self):
        return list(self.dict_models.values())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.is_main_device:
            #logger.info(f"x:\n{x}")
            #logger.info(f"self.dict_models:\n{self.dict_models}")
            logger.info(f"self.dict_models.keys():\n{self.dict_models.keys()}")
        models_outputs = list()
        for name_model in self.dict_models.keys():            
            model = self.dict_models[name_model]
            #if self.is_main_device:
            #    logger.info(f"model:\n{model}")
            #models_outputs.append(output * self.weight_votations[name_model] )
            output = model(x)
            if self.is_main_device:
                #logger.info(f"output:\n{models_outputs}")
                logger.info(f"type(output): {type(output)}")        

                logger.info(f"self.weight_votations: {self.weight_votations}")
                logger.info(f"type(self.weight_votations): {type(self.weight_votations)}")

            output *= self.weight_votations[name_model]
            if self.is_main_device:
                logger.info(f"output - tras producto:\n{output}")
            models_outputs.append(output)

        if self.is_main_device:
            logger.info(f"Models outputs - post weighted:\n{models_outputs}")
            logger.info(f"type(models_outputs]): {type(models_outputs)}")
        
        # Sumamos los resultados ponderados de cada modelo
        # Para hacer las siguientes operaciones como se espera, tiene que ser un Tensor
        #models_outputs = torch.Tensor(models_outputs)
        # Votations
        votations = models_outputs[0]
        if self.is_main_device:
            logger.info(f"type(votations):\n{type(votations)}")
            logger.info(f"votations:\n{votations}")
        for i in range(1, len(models_outputs)):
            votations += models_outputs[i]
            
        # TODO: Revisar esta función 
        # Faltaría: 
        # - ver qué salida quiere en el caso de un clasificador y dársela.
        # - hacer un max de la clase más votada si espera 0 en una 1 en el resto, o, si no, normalizarla.

        if self.is_main_device:
            logger.info(f"type(votations):\n{type(votations)}")
            logger.info(f"Pre-resultado.  Votations results:\n{votations}")        
            #logger.info(f"max(votations):\n{max(votations)}")

        #Normalizamos los valores
        result = torch.empty(votations.size())
        for i in range(len(votations)):
            min_votations = float(min(votations[i]))
            max_votations = float(max(votations[i]))
            if self.is_main_device:
                logger.info(f"min(votations[i]):\n{min_votations}")
                logger.info(f"max(votations[i]):\n{max_votations}")
            #result[i] = (votations[i] - float(min(votations[i]))) / float((max(votations[i]) - min(votations[i])))
            # Los valores deben estar en valores entre 0 y "num_classes"
            result[i] = (self.num_classes - 1) * ((votations[i] - min_votations) / (max_votations - min_votations))

        return result
    
    # TODO: hacer esto bien!
    # TODO: Posible solución, hacer que los parámetros del ensemble sean los de todos sus modelos en el constructor (ir sacándolos y añadiéndolos)
    # TODO: Alternativa. Poner un optimizador para cada modelo y aplicar cada optimizador a cada modelo (?)
    # Para que el optimizador no se queje de que no hay parámetros
    #def parameters(self, recurse = True):   #
    #    #list_tensors_parameters = [torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()]
    #    
    #    print(f"self.dict_models.values():\n{self.dict_models.values()}")
    #    #list_tensors_parameters = [torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()]
    #    list_tensors_parameters = [torch.Tensor(module.parameters(recurse)) for module in self.dict_models.values()]    #
    #    print(f"list_tensors_parameters:\n{list_tensors_parameters}")   #
    #    return torch.concat(list_tensors_parameters)
    #    #return torch.concat([torch.Tensor(list(module.parameters(recurse))) for module in self.dict_models.values()])
    #    #return super().parameters(recurse)
    
