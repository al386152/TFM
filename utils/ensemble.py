import torch

import utils.model_related as mr
from .log_writer import getLogWritter
from .constants import MODEL_TYPE_REGRESSION

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
    # 
    #

    def __init__(self, dict_models:dict, device:torch.device, types_models:dict, 
                 regression_class_boundaries:dict, weight_votations:dict = None,
                 is_main_device:bool = False, num_classes = 5, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if is_main_device:
            logger.info("Creating ensemble")
        self.is_main_device = is_main_device
        self.dict_models = dict_models
        self.weight_votations = weight_votations
        self.num_classes = num_classes
        self.device = device
        self.types_models = types_models
        self.regression_class_boundaries = regression_class_boundaries
        #for _, model in self.dict_models:
        #    # capas_entrenar_final=  1: Se entrena al menos el clasificador. 0, no se entrena nada. -1 se entrena todo.
        #    mr.transfer_learning(model=model, capas_entrenar_final = 1, is_main_device=is_main_device)

        if self.is_main_device:
            logger.info(f"Ensemble created with the following models:\n{dict_models.keys()}")

    def get_list_models(self):
        return list(self.dict_models.values())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # TODO: Ver como arreglar el caso de regresión y cómo tratar cuando hay un modelo de clasificación y uno de regresión.
        #   Contexto: 
        #       - En el caso de la regresión los resultados son un número real.
        #       - En el caso de la regresión, se espera que los valores salgan como un número real para ser transformados.
        #       - En el caso de la clasificación, los resultados son una lista de valores.
        #       - ¿Solución?: 
        #           1) Añadir como parámetro que sea una lista de si es clasificación o regresión
        #           2) Hacer las trasformaciones de la salida aquí (esto es, de regresión a clasificación + pesos votación) y no hacerlas luego


        if self.is_main_device:
            logger.debug(f"x:\n{x}")
            logger.debug(f"self.dict_models.keys():\n{self.dict_models.keys()}")

        models_outputs = list()
        for name_model in self.dict_models.keys():            
            model = self.dict_models[name_model]

            output = model(x).to(self.device)
            if self.is_main_device:
                logger.info(f"self.types_models[{name_model}]:\n{self.types_models[name_model]}\nRegression: {self.types_models[name_model] == MODEL_TYPE_REGRESSION}") # TODO: Convertir en debugs

            if self.types_models[name_model] == MODEL_TYPE_REGRESSION:
                output = mr.from_regression_to_classification(outputs=output, boundaries=self.regression_class_boundaries, 
                                                            num_classes=self.num_classes, device=self.device)
            if self.is_main_device:
                logger.info(f"output: {output}")
                logger.debug(f"type(output): {type(output)}")        
                logger.debug(f"self.weight_votations: {self.weight_votations}")
                logger.debug(f"type(self.weight_votations): {type(self.weight_votations)}")

            if self.is_main_device:
                logger.info(f"output: {output}") # TODO: Borrar

            output *= self.weight_votations[name_model].to(self.device)
            if self.is_main_device:
                logger.debug(f"output - tras producto:\n{output}")
            models_outputs.append(output)

        if self.is_main_device:
            logger.debug(f"Models outputs - post weighted:\n{models_outputs}")
            logger.debug(f"type(models_outputs]): {type(models_outputs)}")
        
        # Sumamos los resultados ponderados de cada modelo
        # Para hacer las siguientes operaciones como se espera, tiene que ser un Tensor
        #models_outputs = torch.Tensor(models_outputs)
        # Votations
        votations = models_outputs[0].to(self.device)
        if self.is_main_device:
            logger.debug(f"type(votations):\n{type(votations)}")
            logger.debug(f"votations:\n{votations}")
        for i in range(1, len(models_outputs)):
            votations += models_outputs[i]

        if self.is_main_device:
            logger.debug(f"type(votations):\n{type(votations)}")
            logger.debug(f"Pre-resultado.  Votations results:\n{votations}")        

        #Normalizamos los valores
        result = torch.empty(votations.size()).to(self.device)
        for i in range(len(votations)):
            min_votations = float(min(votations[i]))
            max_votations = float(max(votations[i]))
            if self.is_main_device:
                logger.debug(f"min(votations[i]):\n{min_votations}")
                logger.debug(f"max(votations[i]):\n{max_votations}")
            result[i] = (votations[i] - min_votations) / (max_votations - min_votations)
        
        if self.is_main_device:
            logger.info(f"result - post-normalize: {result}")

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
    
