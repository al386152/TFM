import torch

import utils.model_related as mr
from .log_writer import getLogWritter
from .constants import MODEL_TYPE_REGRESSION
from .operations import normalize_tensor

logger = getLogWritter(__name__)

class Ensemble(torch.nn.Module):

    # list_models:
    #   ->Tupla: (nombre del modelo, el modelo)
    # 
    # weight_votations: clave: 
    #   -> clave: nombre del modelo
    #   -> valor: diccionario de clases tal que:
    #       -> la clave es el nombre/identificador de la clase
    #       -> el valor es el peso de la votación de ese modelo en esa clase
    def __init__(self, list_models:list, device:torch.device, types_models:dict, 
                 regression_class_boundaries:dict, weight_votations:dict = None,
                 is_main_device:bool = False, num_classes:int = 5, freeze_models_weights:bool = False,
                 create_classifier:bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if is_main_device:
            logger.info("Creating ensemble")
        self.is_main_device = is_main_device
        self.list_models = list_models        
        self.weight_votations = weight_votations
        self.num_classes = num_classes
        self.device = device
        self.types_models = types_models
        self.regression_class_boundaries = regression_class_boundaries        

        self.list_model_names = [nombre for nombre,_ in list_models]

        if freeze_models_weights:
            for model in self.list_models:
                for param in model.parameters():
                    param.requires_grad = False

        if create_classifier:
            # Van a haber tantas entradas como el número de clases (salidas del clasificador de cada modelo) por cada modelo.
            self.classifier = torch.nn.Linear(num_classes * len(self.list_models), num_classes)
        else:
            self.classifier = None
        

        if self.is_main_device:            
            logger.info(f"Ensemble created with the following models:\n{[(self.list_model_names[i], self.types_models[i]) for i in range(len(self.types_models))]}")

    def get_list_models(self):
        return self.list_model_names

    def __str__(self):
        return f"{super().__str__()} - {str([(self.list_model_names[i], self.types_models[i]) for i in range(len(self.types_models))])}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        
        x = x.clone() # Por si acaso se modifica la entrada original

        if self.is_main_device:
            logger.debug(f"self.list_models:\n{self.list_models}")

        # Obtenemos las salidas de todos los modelos
        models_outputs = list()
        for i in range(len(self.list_models)):

            name_model, model = self.list_models[i]

            output = model(x).to(self.device)
            if self.is_main_device:
                logger.debug(f"self.types_models[{i}] ({name_model}) :\n{self.types_models[i]}\n Is regression: {self.types_models[i] == MODEL_TYPE_REGRESSION}")

            # Transformando la salido para que tenga el formato de una de clasificación
            if self.types_models[i] == MODEL_TYPE_REGRESSION:
                if self.is_main_device:
                    logger.debug(f"self.regression_class_boundaries[{i}] {self.regression_class_boundaries[i]}")

                output = mr.from_regression_to_classification(outputs=output, boundaries=self.regression_class_boundaries[i].to(self.device), 
                                                              num_classes=self.num_classes, device=self.device)
            #else: output = output # La salida ya está en el formato de una de clasificación.

            if self.is_main_device:
                logger.debug(f"output: {output}\ntype(output): {type(output)}\nself.weight_votations: {self.weight_votations}\ntype(self.weight_votations): {type(self.weight_votations)}")                                                

            if self.classifier is not None:            
                # Se pondera la salida en función de los pesos y se añade a la lista desde la cual se van a acumular todos.
                output *= self.weight_votations[i].to(self.device)
                if self.is_main_device:
                    logger.debug(f"output - tras producto:\n{output}")
            #else: No se modifican las salidas, el plan es que lo haga el clasificador.
            models_outputs.append(output)

        if self.is_main_device:
            logger.debug(f"Models outputs - post weighted:\n{models_outputs}\ntype(models_outputs]): {type(models_outputs)}")
        
        if self.classifier is not None:
            # Acumulamos las votaciones de cada modelo y los normalizamos
            votations = sum(models_outputs).to(self.device)

            if self.is_main_device:
                logger.debug(f"type(votations):\n{type(votations)}\nPre-normalizado.  Votations results:\n{votations}")

            #Normalizamos los valores        
            normalize_tensor(data=votations, output=(result:=torch.empty(votations.size()).to(self.device)) )
            
            if self.is_main_device:
                logger.debug(f"result - post-normalize: {result}")
        else:
            models_outputs = torch.stack(models_outputs).to(self.device)
            if self.is_main_device:                
                logger.info(f"torch.stack(models_outputs): {models_outputs}") # TODO: convertir en debug

            result = self.classifier(models_outputs)

            if self.is_main_device:                
                logger.info(f"result: {result}") # TODO: convertir en debug

        return result
