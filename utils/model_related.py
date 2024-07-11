import os

import torch

import utils.constants as cons
import utils.metrics as m
from .operations import OUTPUT_MODEL_NAME
from .log_writer import getLogWritter

from torch.nn.parallel import DistributedDataParallel

logger = getLogWritter(__name__)

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

# https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py
def fine_tuning(model, model_name, outputs, is_main_device):
    
    logger.info("Adding fine-tuning layers")                        
    
    #if model_name == "vgg19":
    if "vgg" in model_name:
        classifier_layer = model.classifier[6]        
    #elif model_name == "resnet50":
    elif "resnet" in model_name:
        classifier_layer = model.fc
    elif "densenet" in model_name:
        classifier_layer = model.classifier
    else: 
        logger.warning(f"Han habido varias comprobaciones antes, ¿cómo has llegado aquí?. 'model_name: {model_name}'")
        classifier_layer = None

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

def saving_the_model(args: dict, model):
    model_name = OUTPUT_MODEL_NAME(name=args[cons.MODEL], number_clases=args[cons.NUMBER_CLASSES])
    logger.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)
# -- Fin saving_the_model -- #

# TODO: por completar.
def inference(args: dict, model, dataloaders, device):
    #model.eval()    
    evaluate_model(model=model,dataloader=dataloaders[cons.VALIDATION_FOLDER_NAME], 
                      device=device, lista_metricas=m.get_list_metrics_with_CM())
# -- Fin inference -- #

def evaluate_model(model, dataloader, device, is_main_device, lista_metricas: list):
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            m.update_metrics(lista_metricas, outputs=outputs, labels=labels)
    
    lista_resultados = m.get_metrics()
    
    if is_main_device:
        for name, metric in lista_resultados:
            logger.info(f'{name}: {metric:.4f}' if name != "ConfusionMatrix" else f"{name}:\n{metric}")

    m.reset_list_metrics(lista_metricas) 

    return lista_resultados
# -- Fin evaluate_model -- #


def load_model(args: dict, device, is_main_device, fine__tuning:bool = True) -> torch.nn.Module:

    model = None
    model_name = args[cons.MODEL]
    model_weights_path = args[cons.MODEL_WEIGHTS]
    outputs = args[cons.NUMBER_CLASSES]

    if is_main_device:
        logger.debug(f"model_weights_path: {model_weights_path}")

    if model_weights_path and os.path.isfile(model_weights_path):        
        model_weights_path = model_weights_path
        weights = None
    else: 
        weights = cons.DEFAULT_MODEL_WEIGHTS

    if is_main_device:
        logger.info(f"Cargando los siguientes pesos: \'{weights if weights is not None else model_weights_path}\'")

    model = cons.SWITCH_MODELOS[model_name](weights = weights)

    if fine__tuning: 
        model = fine_tuning(model=model, model_name=model_name,outputs=outputs, is_main_device=is_main_device)
                
    if model_weights_path:
        # Es importante cargarlo *DESPUES* del fine tuning
        model.load_state_dict(torch.load(model_weights_path))        
    
    model = transfer_learning(model, args[cons.NOT_FREEZE_LAYERS], is_main_device)

    model = model.to(device)
    if args[cons.IS_DISTRIBUTED]:
        model = DistributedDataParallel(model, device_ids=[device])

    return model
# -- Fin load_model -- #

