import os

import torch

import utils.constants as cons
import utils.metrics as m
from .operations import OUTPUT_MODEL_NAME
from .log_writer import getLogWritter
from .ensemble import Ensemble
from torch.nn.parallel import DistributedDataParallel

logger = getLogWritter(__name__)

# TODO: Hacer esto bien
# Esto lo estoy poniendo esto de esta forma por simplicidad y para estos casos.
def modify_model_layers(model:torch.nn.Module, model_name:str, args:dict):

    mod_classifier_layer = list()

    classifier_layer = get_classifier_layer(model, model_name)
    num_ftrs = classifier_layer.in_features
    # TODO: Poner una variable para que controle esto
    mod_classifier_layer.append(torch.nn.BatchNorm1d(num_features=num_ftrs))    
    mod_classifier_layer.append(torch.nn.LayerNorm(normalized_shape=num_ftrs))    

    if "resnet152" == model_name:
        
        mod_classifier_layer.append(torch.nn.Dropout(p = args[cons.P_DROPOUT]))
        mod_classifier_layer.append(model.fc)
        model.fc = torch.nn.Sequential(*mod_classifier_layer)

    elif "densenet121" == model_name:

        mod_classifier_layer.append(torch.nn.Dropout(p = args[cons.P_DROPOUT]))
        mod_classifier_layer.append(model.classifier)
        model.classifier = torch.nn.Sequential(*mod_classifier_layer)
# -- FIN modify_model_layers -- #

def transfer_learning(model:torch.nn.Module, capas_entrenar_final:int = -1, is_main_device = True):

    if capas_entrenar_final == -1:
        return model
    # else: se congelan las capas ==>
    if is_main_device:
        _info = f"s {capas_entrenar_final} últimas" if capas_entrenar_final > 1 else " última"
        logger.info(f"Preparando congelar todas las capas menos la {_info} para el transfer learning")
    
    capas_congelar = list(model.children())[: -1 * capas_entrenar_final] if capas_entrenar_final > 0 else model.children()

    for child in capas_congelar:
        if is_main_device:
            logger.debug(f"{str(child)}")
        for param in child.parameters():
            param.requires_grad = False   

    if is_main_device:
        logger.info(f"{'-'*4} Fin función transfer learning {'-'*4}")

    return model
# -- Fin transfer_learning -- #

def get_classifier_layer(model: torch.nn.Module, model_name: str, is_main_device: bool=False)->torch.nn.modules.linear.Linear:
    #print(f"model_name: {model_name}")
    if is_main_device: 
            logger.info(f"model_name: {model_name}")
    if "vgg" in model_name:
        classifier_layer = model.classifier[6]        
    #elif model_name == "resnet50":
    elif "resnet" in model_name:
        classifier_layer = model.fc
    elif "densenet" in model_name:
        classifier_layer = model.classifier
    else:        
        if is_main_device: 
            logger.warning(f"Han habido varias comprobaciones antes, ¿cómo has llegado aquí?. 'model_name: {model_name}'")
        classifier_layer = None

    return classifier_layer
# -- Fin get_classifier_layer -- #    

# https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py
def fine_tuning(model, model_name, outputs, is_main_device):
    
    if is_main_device:
        logger.info("Adding fine-tuning layers")                        
    
    classifier_layer = get_classifier_layer(model, model_name, is_main_device)

    num_ftrs = classifier_layer.in_features
    if is_main_device:
        logger.debug(f"num_ftrs: {str(num_ftrs)}")            

    # Parece ser que así no se guarda bien, al revés, crea una capa más en vez de sustituir el clasificador
    #model.classifier_layer = torch.nn.Sequential( torch.nn.Linear(num_ftrs, outputs), torch.nn.LayerNorm(outputs) )    
    # model.classifier_layer = torch.nn.Linear(num_ftrs, outputs)

    capa_clasificacion = torch.nn.Linear(num_ftrs, outputs)
    if "vgg" in model_name:
        model.classifier[6] = capa_clasificacion
    elif "resnet" in model_name:
        model.fc = capa_clasificacion
    elif "densenet" in model_name:
        model.classifier = capa_clasificacion
    # else: No se debería dar el caso

    return model
# -- Fin fine_tuning -- #

def saving_the_model(args: dict, model: torch.nn.Module):
    # Esta función se tiene que ejecutar solo en un único hilo.
    model_name = OUTPUT_MODEL_NAME(name=args[cons.MODEL], number_clases=args[cons.NUMBER_CLASSES], 
                                   is_regression=args[cons.IS_REGRESSION])    
    logger.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.module.state_dict() if args[cons.IS_DISTRIBUTED] else model.state_dict(),
               model_name)
# -- Fin saving_the_model -- #


def from_regression_to_classification(outputs:torch.Tensor, boundaries:torch.Tensor, num_classes:int, device):

    # Se ponen los datos en la clase que les tocaría
    outputs = torch.bucketize(input=outputs, boundaries=boundaries)    
    outputs = outputs.to(torch.int)

    # Se preparan los datos en el formato esperado para las métricas.
    addapted_output = list()

    for ouput in outputs:
        mod_ouput = [0] * num_classes
        # Al parecer, hay algún caso en el que se pasa a una clase inexistente
        clase = ouput if ouput < num_classes else num_classes - 1
        mod_ouput[clase] = 1
        addapted_output.append(mod_ouput)

    addapted_output = torch.Tensor(addapted_output)

    return addapted_output.to(device)


def evaluate_model(model, dataloader, device, is_main_device, lista_metricas: list, args, loss_fn=None, save_confusion_matrix:bool=True, 
                   nombre_prueba:str="Test", metricas_regression:list=None):
    
    if is_main_device:
        logger.debug(f"evaluate_model - inicio")
    
    num_elementos = len(dataloader)
    log_info_cada = int(num_elementos * cons.TANTO_POR_UNO_LOGS_PRINT) if int(num_elementos * cons.TANTO_POR_UNO_LOGS_PRINT) > 0 else 1
    running_val_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):

            if is_main_device and (i % log_info_cada == 0):
                info_print = f"{nombre_prueba} - Batch: [{i + 1}/{num_elementos}]"
                logger.info(info_print)

            inputs, labels = inputs.to(device), labels.to(device)

            if is_main_device:
                logger.debug(f"Inputs shape: {inputs.shape}, min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.mean()}")
                logger.debug(f"Labels shape: {labels.shape}, labels: {labels}")

            outputs = model(inputs)            
    
            outputs = outputs.to(device)
            
            if args[cons.IS_REGRESSION]:
                outputs = outputs.reshape(labels.shape).to(device)

            if is_main_device:
                logger.debug(f"{nombre_prueba} Outputs shape: {outputs.shape}, {nombre_prueba} Labels shape: {labels.shape}")
                logger.debug(f"Outputs shape: {outputs.shape}, Outputs: {outputs}")
                
            if loss_fn != None:
                loss = loss_fn(outputs, labels)
                running_val_loss += loss
            
            if args[cons.IS_REGRESSION]:
                if is_main_device:
                    logger.debug(f"boundaries: {args[cons.REGRESSION_CLASS_BOUNDARIES]}")

                regression_outputs = outputs.reshape(labels.shape).to(device)
                m.update_metrics(metricas_regression, outputs=regression_outputs, labels=labels)
                
                outputs = from_regression_to_classification(outputs=outputs, 
                                                            boundaries=(args[cons.REGRESSION_CLASS_BOUNDARIES][cons.BOUNDARIES_ENSEMBLE_REGRESSION_CLASSIFIER].to(device=device)),
                                                            num_classes=args[cons.NUMBER_CLASSES], device=device)
                
            m.update_metrics(lista_metricas, outputs=outputs, labels=labels)

    if is_main_device:
        logger.debug(f"evaluate_model - get metrics")

    dict_resultados = m.get_metrics(lista_metricas, args=args, save_confusion_matrix=save_confusion_matrix, is_main_device=is_main_device)        

    if loss_fn != None:
        dict_resultados[cons.AVG_LOSS] = running_val_loss / (i + 1) 
        dict_resultados[cons.EPOCH_LOSS] = running_val_loss

    if is_main_device:
        logger.debug(f"dict_resultados:\n{dict_resultados}")
        for name in dict_resultados:            
            logger.info(f"{name}:\n{dict_resultados[name]}" if (name == cons.CONFUSION_MATRIX or name == cons.NORM_CONFUSION_MATRIX)  \
                        else f"{name}: {dict_resultados[name]:.4f}")

        if args[cons.IS_REGRESSION]:
            dict_resultados_regresion = m.get_metrics(metricas_regression, args=args, save_confusion_matrix=save_confusion_matrix, 
                                                        is_main_device=is_main_device)        
            for name in dict_resultados_regresion:
                logger.info(f"{name}:\n{dict_resultados[name]}")
        
            m.reset_list_metrics(metricas_regression)

    m.reset_list_metrics(lista_metricas)

    return dict_resultados
# -- Fin evaluate_model -- #

def load_model_weights(args: dict, model_name: str, model: torch.nn.Module, device:torch.device,  model_weights_path:str, is_main_device:bool):
    if is_main_device:
            logger.info(f"----load_model_weights----")
            logger.debug(f"model_weights_path: {model_weights_path}\n ----")
            logger.debug(f"device: {device}\n ----")
            logger.debug(f"type(device): {type(device)}")
            logger.debug(f"module.state_dict():\n{model.state_dict().keys()}")

    state_dict = torch.load(f=model_weights_path, weights_only=args[cons.INFERENCE])#,
                            #map_location=device)
    if is_main_device:
        logger.debug(f"state_dict:\n{state_dict.keys()}")    

    model.load_state_dict(state_dict, strict=True)
# -- Fin load_model_weights -- #

def load_model(args: dict, device, is_main_device:bool, fine__tuning:bool = True) -> torch.nn.Module:

    model = None
    model_names = args[cons.MODEL]
    model_weights_path = args[cons.MODEL_WEIGHTS]
    models_types = args[cons.ENSEMBLE_TYPE_MODEL]

    if is_main_device:
        logger.info(f"model_weights_path: {model_weights_path}")

    if is_main_device:
        logger.info(f"Cargando los siguientes pesos: \'{cons.DEFAULT_MODEL_WEIGHTS if model_weights_path is None else model_weights_path}\'")

    if is_main_device:
        logger.info(f"models_types: \'{models_types}\'") # TODO: Convertirlo en debug

    list_models = list()
    for i in range(len(model_names)):
        model_name = model_names[i]
        
        if is_main_device:
            logger.info(f"models_types[{i}]: \'{models_types[i]}\'") # TODO: Convertirlo en debug

        outputs = args[cons.NUMBER_CLASSES] if models_types[i] != cons.MODEL_TYPE_REGRESSION else 1
        if is_main_device:
            logger.info(f"Cargando el modelo: {model_name}")        
        
        if model_weights_path and os.path.isfile(model_weights_path[i]):
            model = cons.SWITCH_MODELOS[model_name]()
        else: 
            model = cons.SWITCH_MODELOS[model_name](weights = cons.DEFAULT_MODEL_WEIGHTS)     

        if is_main_device:
            logger.debug(f"Modelo cargado: {model}")             

        if fine__tuning: 
            model = fine_tuning(model=model, model_name=model_name, outputs=outputs, is_main_device=is_main_device)     

        if is_main_device:
            logger.info(f"modify_model_layers")        
        modify_model_layers(model=model, model_name=model_name, args=args)
        if is_main_device:
            logger.debug(f"model.to(device)")
        model = model.to(device)

        #Se pueden cargar pesos de un modelo o de "DistributedDataParallel". 
        # Ambos son iguales, pero se modifican las claves del diccionario de pesos, 
        # Entonces, si no puede cargarlos como el propio modelo, lo intenta como DistributedDataParallel
        try:
            if is_main_device:
                logger.info("Tratando de cargar los pesos en el modelo base")
            if model_weights_path:
                load_model_weights(args=args, model_name=model_name, model=model, 
                                device=(torch.cuda.device(device) if "cuda" in args[cons.DEVICE] else torch.device("cpu")),
                                model_weights_path=model_weights_path[i], is_main_device=is_main_device)
            pesos_por_cargar = False
            if is_main_device:
                logger.info("Pesos cargados en el modelo base")
        except:
            if is_main_device:
                logger.info("Los pesos no se han cargado en el modelo base ==> se intentarán cargar como DistributedDataParallel")
            pesos_por_cargar = True
        
        if args[cons.IS_DISTRIBUTED]:
            if is_main_device:
                logger.info(f"DistributedDataParallel")
            model = DistributedDataParallel(model, device_ids=[device])
        
        if model_weights_path and pesos_por_cargar:
            load_model_weights(args=args, model_name=model_name, model=model, 
                               device=(torch.cuda.device(device) if "cuda" in args[cons.DEVICE] else torch.device("cpu")),
                               model_weights_path=model_weights_path[i], is_main_device=is_main_device)
            
        model = transfer_learning(model, args[cons.NOT_FREEZE_LAYERS], is_main_device)
        list_models.append((model_name, model))

    if len(model_names) > 1:        
        model = Ensemble(list_models=list_models, weight_votations=args[cons.ENSEMBLE_VOTATION_WEIGHTS],
                         types_models=args[cons.ENSEMBLE_TYPE_MODEL], 
                         regression_class_boundaries= args[cons.REGRESSION_CLASS_BOUNDARIES],
                         create_classifier=args[cons.ENSEMBLE_CREATE_CLASSIFIER],
                         num_outpus= 1 if args[cons.IS_REGRESSION] else args[cons.NUMBER_CLASSES],
                         num_classes=args[cons.NUMBER_CLASSES],
                         is_main_device=is_main_device, device=device)
        
        if args[cons.IS_DISTRIBUTED]:
            if is_main_device:
                logger.info(f"DistributedDataParallel")
            model = DistributedDataParallel(model, device_ids=[device])
    else:
        _, model = list_models[0]

    return model
# -- Fin load_model -- #
