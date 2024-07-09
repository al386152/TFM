import os

from typing import *
from datetime import datetime

import torch.utils
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch import distributed, cuda
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import utils.arguments_parser as ap
import utils.constants as cons
from utils.common_operations import OUTPUT_MODEL_NAME, GET_TIME_HMS_FORMAT

import utils.model_related as mr
from utils.log_writer import getLogWritter
from utils.common_operations import GET_TIME_HMS_FORMAT, MEAN_TIMES, MULTIPLY_TIME

# Esto es para tener el logger
logger = getLogWritter(__name__)

def load_model(args: dict, device, is_main_device, fine_tuning:bool = True) -> torch.nn.Module:

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

    if fine_tuning: 
        model = mr.fine_tuning(model=model, model_name=model_name,outputs=outputs, is_main_device=is_main_device)
                
    if model_weights_path:
        # Es importante cargarlo *DESPUES* del fine tuning
        model.load_state_dict(torch.load(model_weights_path))        
    
    model = mr.transfer_learning(model, args[cons.NOT_FREEZE_LAYERS], is_main_device)

    model = model.to(device)
    if args[cons.IS_DISTRIBUTED]:
        model = DistributedDataParallel(model, device_ids=[device])

    return model

def load_datasets(args:dict, is_main_device) -> dict:

    transform = transforms.Compose([
        # TODO: MODIFICAR para hacer el aumento de datos aquí
        transforms.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),
        transforms.ToTensor()
    ])

    if args[cons.SHOW_DEBUG_OUTPUTS] and is_main_device:
        logger.debug("load_datasets")
        dict_datasets = dict()

        for folder_name in cons.LIST_FOLDER_NAMES:
            if is_main_device:
                logger.debug(f"folder name: {folder_name}. Data path: {args[cons.DATA_PATH]}.")
            path = os.path.join(args[cons.DATA_PATH], folder_name)
            if is_main_device:
                logger.debug(f"path: {path}")
            dict_datasets[folder_name] = datasets.ImageFolder(path, transform = transform)             

        for data_set in dict_datasets:
            logger.debug(f"{data_set}:\n\t{dict_datasets[data_set]}")

        return dict_datasets
    else:
        return {
            folder_name:
            datasets.ImageFolder(root = os.path.join(args[cons.DATA_PATH], folder_name),
                                transform = transform) 
            for folder_name in cons.LIST_FOLDER_NAMES
        }    

def load_samplers(args:dict, datasets: Tuple)-> dict:    

    dict_samplers = {        
        #folder_name: torch.utils.data.SequentialSampler(folder_name)
        folder_name:
            ( DistributedSampler(datasets[cons.TRAIN_FOLDER_NAME], shuffle=True) if args[cons.IS_DISTRIBUTED] else
                torch.utils.data.SequentialSampler(datasets[folder_name]) )
        for folder_name in cons.LIST_FOLDER_NAMES
    }
    
    #if args[cons.IS_DISTRIBUTED]: dict_samplers[cons.TRAIN_FOLDER_NAME] = DistributedSampler(datasets[cons.TRAIN_FOLDER_NAME], shuffle=True)

    return dict_samplers

# TODO: Por comprobar de que está bien
def load_data_loaders(args: dict, datasets:dict) -> dict:

    samplers = load_samplers(args=args, datasets=datasets)

    return {
        folder_name : DataLoader(
            datasets[folder_name], 
            sampler=samplers[folder_name],
            batch_size=args[cons.BATCH_SIZE],
            drop_last=False,
            pin_memory=False,
            
            # shuffle = True # Con el Sampler debería hacerse automáticamente.
        )
        for folder_name in cons.LIST_FOLDER_NAMES
    }

def train_model(args: dict, model, dataloaders, is_main_device, device, lista_metricas):
    
    if is_main_device:
        logger.info( ('-' * cons.NUM_GUIONES) + "Starting to train the model" + ('-' * cons.NUM_GUIONES) )

    # Preparando las variables
    best_vloss = float('inf') # Número imposible para que en la primera iteración sea menor sí o sí.
    partial_models_path = args[cons.PARTIAL_MODELS_PATH] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = mr.EarlyStopper(patience=3, min_delta=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Creando carpetas para las salidas  
    if not os.path.isdir(partial_models_path):
        os.makedirs(partial_models_path)

    # Entrenando
    if is_main_device:
        tiempos = list()
        estimacion_fin = "(Tiempo estimado por época)"
        estimacion_fin_todos = "(Tiempo estimado total restante)"

    num_epochs = args[cons.EPOCS]

    for epoch in range(num_epochs):       
        
        if is_main_device:
            tiempo_inicio = datetime.now()
            info_print = f"Epoch: [{epoch}/{num_epochs}] - {estimacion_fin} || {estimacion_fin_todos}"
            logger.info(info_print)

        model.train(True)
        avg_loss = mr.train_one_epoch(model=model, epoch_index=epoch, 
                                    training_loader=dataloaders[cons.TRAIN_FOLDER_NAME], 
                                    loss_func=loss_fn, optimizer=optimizer)
        running_val_loss = 0.0

        model.eval()
            
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(dataloaders[cons.VALIDATION_FOLDER_NAME]):                                
                vinputs, vlabels = vdata

                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                voutputs = model(vinputs)                
                vloss = loss_fn(voutputs, vlabels)
                running_val_loss += vloss
                
                if is_main_device:
                    logger.debug(f"vinputs:\n{str(vinputs)}" )
                    logger.debug(f"vlabels:\n{str(vlabels)}" )
                    logger.debug(f"voutputs:\n{str(voutputs)}" )
                
                mr.update_metrics(lista_metricas, outputs=voutputs, labels=vlabels)

        #mr.get_metrics()
        mr.reset_list_metrics(lista_metricas)

        avg_val_loss = running_val_loss / (i + 1)
        if is_main_device:
            logger.info(f"Avg.loss: {avg_loss} | Avg.validation loss: {avg_val_loss}")                    

        if is_main_device and avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_name = f"model_{epoch}_{timestamp}.pth"
            model_path = os.path.join(args[cons.PARTIAL_MODELS_PATH], 
                                    model_name) 

            torch.save(model.state_dict(), model_path)

        # Para el early stopping
        if early_stopper.early_stop(running_val_loss):
            logger.info(f"Stopping the training. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")            
            break
        
        if is_main_device:
            logger.debug(f"tiempo_inicio: {str(tiempo_inicio)}")
            ahora = datetime.now()
            #tiempos.append(datetime.now() - tiempo_inicio)
            logger.debug(f"ahora: {str(ahora)}")
            tiempos.append(ahora - tiempo_inicio)
            #_estimacion_fin = sum(tiempos)/len(tiempos)
            _estimacion_fin = MEAN_TIMES(tiempos)
            logger.debug(f"_estimacion_fin: {str(_estimacion_fin)}")

            _estimacion_fin_todos = MULTIPLY_TIME(_estimacion_fin, (num_epochs - i) )
            logger.debug(f"_estimacion_fin_todos: {str(_estimacion_fin_todos)}")
            estimacion_fin = GET_TIME_HMS_FORMAT(_estimacion_fin)
            estimacion_fin_todos = GET_TIME_HMS_FORMAT(_estimacion_fin_todos)
    
    if is_main_device:
        logger.info(('-' * cons.NUM_GUIONES) + " Training ended " + ('-' * cons.NUM_GUIONES))
    

def saving_the_model(args: dict, model):
    model_name = OUTPUT_MODEL_NAME(name=args[cons.MODEL], number_clases=args[cons.NUMBER_CLASSES])
    logger.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)

def inference(args: dict, model, dataloaders, device):
    #model.eval()
    mr.evaluate_model(model=model,dataloader=dataloaders[cons.VALIDATION_FOLDER_NAME], 
                      device=device, num_classes=args[cons.NUMBER_CLASSES])

def setup_gpu(args:dict):

    # Estableciendo el dispositivo en el que se va a trabajar.
    device = "cpu"
    worldsize = 1
    if "cuda" in args[cons.DEVICE]:
        logger.debug(f"cuda in {str(args[cons.DEVICE])}")
        if not torch.cuda.is_available():
            logger.info("'torch.cuda' is not available ==> cpu")            
            #device = "cpu"
        else:
            if args[cons.IS_DISTRIBUTED]:
                logger.info("Setting up distributed gpu") # TODO: pensar en un mensaje mejor para el log
                distributed.init_process_group(backend=cons.BACKEND)
                device = int(os.environ["LOCAL_RANK"])
                worldsize = int(os.environ["WORLD_SIZE"])                
            else:
                logger.info("Setting one gpu") # TODO: pensar en un mensaje mejor para el log
                device = args[cons.DEVICE]
    #else: device = "cpu"
    
    if args[cons.IS_DISTRIBUTED]:
        cuda.set_device(device)
    else:
        device = torch.device(device)

    return device, worldsize

def main(args: dict):    

    device, _ = setup_gpu(args=args)

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device:
        logger.info(f"Device: {device}")

    datasets = load_datasets(args, is_main_device)
    if is_main_device:
        logger.debug("\n".join([f"len(dataset): {len(dataset)}\ndataset:\n{dataset}" for dataset in datasets]))

    data_loaders = load_data_loaders(args, datasets)
    if is_main_device:
        logger.debug("\n".join([f"len(dataloader): {len(dataloader)}\dataloader:\n{dataloader}" for dataloader in data_loaders]))

    metricas = mr.get_list_metrics(args[cons.NUMBER_CLASSES])

    model = load_model(args=args, device=device, is_main_device=is_main_device)
    if is_main_device:
        logger.info(f"Model: {model}")

    # if is_main_device:
        t_inicio = datetime.now()
    train_model(args, model=model, dataloaders=data_loaders, is_main_device=is_main_device, device=device, lista_metricas=metricas)
    if is_main_device:
        tiempo_entrenamiento = GET_TIME_HMS_FORMAT((datetime.now() - t_inicio))
        logger.info(f"Tiempo entrenamiento: {tiempo_entrenamiento}")            
    
    if is_main_device:
        saving_the_model(args, model)
        mr.evaluate_model(model=model,dataloader=data_loaders[cons.VALIDATION_FOLDER_NAME], 
                          device=device, is_main_device=is_main_device, num_classes=args[cons.NUMBER_CLASSES])

# Esto lo hago así porque no se me ha ocurrido de otra forma de poner el nivel correcto a todos los loggers de todos los scripts
def poner_nivel_a_todos_loggers():
    lista_loggers = [ap.logger, mr.logger, logger]

    for _logger in lista_loggers:
        _logger.setLevel(cons.loggin_level)

if __name__ == "__main__":
    args = ap.get_dict_args()
    poner_nivel_a_todos_loggers()   
    main(args)
