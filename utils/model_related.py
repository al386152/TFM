import os

import torch

import utils.constants as cons
import utils.metrics as m
from .operations import OUTPUT_MODEL_NAME
from .log_writer import getLogWritter
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
    #if model_name == "vgg19":
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

def saving_the_model(args: dict, model):
    # Esta función se tiene que ejecutar solo en un único hilo.

    model_name = OUTPUT_MODEL_NAME(name=args[cons.MODEL], number_clases=args[cons.NUMBER_CLASSES], 
                                   is_regression=args[cons.IS_REGRESSION])    
    logger.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)
# -- Fin saving_the_model -- #


def from_regression_to_classification(outputs:torch.Tensor, boundaries:torch.Tensor, num_classes:int, device):

    #print(f"outputs - pre | type: {outputs.dtype} |\n{outputs}")
    # Se ponen los datos en la clase que les tocaría
    outputs = torch.bucketize(input=outputs, boundaries=boundaries)    
    outputs = outputs.to(torch.int)
    # Al parecer, hay algún caso en el que se pasa a una clase inexistente
    # apply_ solo se puede hacer en tensores en la CPU :)
    #outputs.apply_(lambda x: (x if x < num_classes else num_classes-1) )
    #print(f"outputs - post | type: {outputs.dtype} |\n{outputs}")

    # Se preparan los datos en el formato esperado para las métricas.
    addapted_output = list()

    #print(f"ouput[0]: {ouput[0]}")
    for ouput in outputs:        
        mod_ouput = [0] * num_classes
        # Al parecer, hay algún caso en el que se pasa a una clase inexistente, como explico arriba.
        clase = ouput[0] if ouput[0] < num_classes else num_classes - 1
        mod_ouput[clase] = 1
        addapted_output.append(mod_ouput)

    #return torch.Tensor(addapted_output)
    addapted_output = torch.Tensor(addapted_output)#.to(torch.int)
    #print(f"addapted_output | type: {addapted_output.dtype} | size: {addapted_output.size()} |\n{addapted_output}")    

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

            #if args[cons.IS_REGRESSION]:
            #    #labels = labels.float()
            #    labels = torch.squeeze(labels)
            #    #labels = labels.to(torch.int)

            inputs, labels = inputs.to(device), labels.to(device)

            if is_main_device:
                logger.debug(f"Inputs shape: {inputs.shape}, min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.mean()}")
                logger.debug(f"Labels shape: {labels.shape}, labels: {labels}")

            outputs = model(inputs)            
    
            outputs = outputs.to(device)

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
                outputs = from_regression_to_classification(outputs=outputs, boundaries=args[cons.REGRESSION_CLASS_BOUNDARIES], 
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

        # TODO: Revisar por qué no funciona bien las métricas de la regresión.
        if args[cons.IS_REGRESSION]:
            dict_resultados_regresion = m.get_metrics(metricas_regression, args=args, save_confusion_matrix=save_confusion_matrix, 
                                                        is_main_device=is_main_device)        
            #print(f"dict_resultados_regresion: {dict_resultados_regresion}")
            for name in dict_resultados_regresion:
                logger.info(f"{name}:\n{dict_resultados[name]}")
        
            m.reset_list_metrics(metricas_regression)

    m.reset_list_metrics(lista_metricas)

    return dict_resultados
# -- Fin evaluate_model -- #


def load_model(args: dict, device, is_main_device, fine__tuning:bool = True) -> torch.nn.Module:

    model = None
    model_name = args[cons.MODEL]
    model_weights_path = args[cons.MODEL_WEIGHTS]
    outputs = args[cons.NUMBER_CLASSES] if not args[cons.IS_REGRESSION] else 1

    if is_main_device:
        logger.info(f"model_weights_path: {model_weights_path}")

    if is_main_device:
        logger.info(f"Cargando los siguientes pesos: \'{cons.DEFAULT_MODEL_WEIGHTS if model_weights_path is None else model_weights_path}\'")

    if model_weights_path and os.path.isfile(model_weights_path):        
        model = cons.SWITCH_MODELOS[model_name]()
    else: 
        model = cons.SWITCH_MODELOS[model_name](weights = cons.DEFAULT_MODEL_WEIGHTS)                
    
    #model = model.to(device)
    #if args[cons.IS_DISTRIBUTED]: model = DistributedDataParallel(model, device_ids=[device])

    if fine__tuning: 
        model = fine_tuning(model=model, model_name=model_name, outputs=outputs, is_main_device=is_main_device)     

    if is_main_device:
        logger.info(f"modify_model_layers")
    # TODO: Hacer bien
    modify_model_layers(model=model, model_name=args[cons.MODEL], args=args)
    if is_main_device:
        logger.info(f"model.to(device)")
    model = model.to(device)
    if args[cons.IS_DISTRIBUTED]:
        if is_main_device:
            logger.info(f"DistributedDataParallel")
        model = DistributedDataParallel(model, device_ids=[device])   

    if model_weights_path:
        if is_main_device:
            logger.info(f"{model_weights_path}\n ----")
            logger.info(f"module.state_dict():\n{model.state_dict().keys()}")

        state_dict = torch.load(model_weights_path, weights_only=args[cons.INFERENCE])

        if is_main_device:
            logger.debug(f"state_dict:\n{state_dict.keys()}")    

        # Al menos con ResNet, esto es necesario
        # Básicamente, no me guarda los pesos del clasificador bien, solo guarda la de la última capa ya que, entiendo, las otras 3 no son necesarias ==> tengo que poner a mano las cosas.
        if "resnet50" == model_name:
            if is_main_device:
                logger.debug('if "resnet50" in model_name:')
        
        elif "densenet169" == model_name:
            if is_main_device:
                logger.debug('if "densenet169" in model_name:')

        model.load_state_dict(state_dict, strict=True )
        if is_main_device and "resnet50" == model_name:
            logger.info(f"module.fc: {model.module.fc}")
            
    model = transfer_learning(model, args[cons.NOT_FREEZE_LAYERS], is_main_device)

    return model
# -- Fin load_model -- #
