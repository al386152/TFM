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
                 is_main_device:bool = False, num_classes:int = 5,
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

        # Se congelan los pesos de los modelos ya preentrenados
        for _,model in self.list_models:
            for param in model.parameters():
                param.requires_grad = False

        if create_classifier:
            # TODO: Hacer que sea la salida del clasificador sin transformar los datos de los modelos de regresión
            # Van a haber tantas entradas como el número de clases (salidas de los clasificadores) por cada modelo.
            num_modelos = len(self.list_models)
            num_regresion = len(list(filter(lambda model: model == MODEL_TYPE_REGRESSION, self.types_models)))
            # Los modelos de clasificación tienen tantas salidas como número de clases, pero los de regresión solo tienen una.
            num_entradas = num_classes * (num_modelos - num_regresion) + num_regresion
            
            if self.is_main_device:            
                logger.info(f"num_modelos: {num_modelos}, num_regresion: {num_regresion}\nself.types_models: {self.types_models}") # TODO: Borrar
                logger.info(f"in_features: {num_entradas}, out_features: {num_classes}")


            self.classifier = torch.nn.Linear(in_features=num_entradas, out_features=num_classes, device=self.device)
        else:
            self.classifier = None
        

        if self.is_main_device:            
            logger.info(f"Ensemble created with the following models:\n{[(self.list_model_names[i], self.types_models[i]) for i in range(len(self.types_models))]}")
    # -- End __init__ -- #

    # Aunque Code diga que no, se utiliza en "trining.py"
    def get_list_models(self):
        return self.list_models
    # -- End get_list_models -- #

    def __str__(self):
        return f"{super().__str__()} - {str([(self.list_model_names[i], self.types_models[i]) for i in range(len(self.types_models))]).replace('[', '').replace(']', '')}"
    # -- End __str__ -- #

    def _forward_no_classifier(self, x: torch.Tensor) -> torch.Tensor:

        models_outputs = list()
        for i in range(len(self.list_models)):

            name_model, model = self.list_models[i]

            output = model(x).to(device=self.device)
            if self.is_main_device: logger.debug(f"self.types_models[{i}] ({name_model}) :\n{self.types_models[i]}\n Is regression: {self.types_models[i] == MODEL_TYPE_REGRESSION}")

            # Transformando la salido para que tenga el formato de una de clasificación
            if self.types_models[i] == MODEL_TYPE_REGRESSION:
                if self.is_main_device:
                    logger.debug(f"self.regression_class_boundaries[{i}] {self.regression_class_boundaries[i]}")

                output = mr.from_regression_to_classification(outputs=output, boundaries=self.regression_class_boundaries[i], 
                                                              num_classes=self.num_classes, device=self.device)
            #else: output = output # La salida ya está en el formato de una de clasificación.

            if self.is_main_device:
                logger.debug(f"output: {output.size()}") # TODO: Convertir en debug
                logger.debug(f"output: {output}\ntype(output): {type(output)}\nself.weight_votations: {self.weight_votations}\ntype(self.weight_votations): {type(self.weight_votations)}")

                     
            # Se pondera la salida en función de los pesos y se añade a la lista desde la cual se van a acumular todos.
            output *= self.weight_votations[i]
            if self.is_main_device: logger.debug(f"output - tras producto:\n{output}")
            #else: No se modifican las salidas, el plan es que lo haga el clasificador.
            models_outputs.append(output.to(device=self.device))

        if self.is_main_device: logger.debug(f"Models outputs - post weighted:\n{models_outputs}\ntype(models_outputs]): {type(models_outputs)}")
        
        # Acumulamos las votaciones de cada modelo y los normalizamos
        votations = sum(models_outputs).to(device=self.device)

        if self.is_main_device: logger.debug(f"type(votations):\n{type(votations)}\nPre-normalizado.  Votations results:\n{votations}")

        #Normalizamos los valores        
        normalize_tensor(data=votations, output=(result:=torch.empty(votations.size()).to(device=self.device)) )
        
        if self.is_main_device: logger.debug(f"result - post-normalize: {result}")

        return result
    # -- End _forward_no_classifier -- #

    def _forward_with_classifier(self, x: torch.Tensor) -> torch.Tensor:
        
        # Generamos la salida de todos los modelos y los metemos todos en un único tensor
        # stack pone un tensor luego del otro, pero queremos que todas las salidas estén en la misma fila.
        #models_outputs = torch.stack([model(x).to(self.device)  for _, model in self.list_models]).to(device=self.device)
        models_outputs = torch.concat([model(x).to(self.device)  for _, model in self.list_models], dim=1).to(device=self.device)
    
        if self.is_main_device:                
            logger.debug(f"torch.stack(models_outputs).size(): {models_outputs.size()}") #convertir en debug
            logger.debug(f"torch.stack(models_outputs): {models_outputs}")

        result = self.classifier(models_outputs).to(device=self.device)

        if self.is_main_device:             
            logger.debug(f"result.size(): {result.size()}") # TODO: convertir en debug   
            logger.debug(f"result: {result}")            
    
        return result
    # -- End _forward_with_classifier -- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        
        x = x.to(device=self.device, copy=True) # Por si acaso se modifica la entrada original, se copia (además de pasarla a la GPU)

        return (self._forward_no_classifier(x) if self.classifier is None 
                else self._forward_with_classifier(x)).to(device=self.device)
    # -- End forward -- #
# -- End Ensemble class -- #
