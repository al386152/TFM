import os

from typing import *
from datetime import datetime

import torch.utils
import torch
from torchvision import datasets, transforms

from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

import utils.arguments_parser as ap
import utils.constants as cons
import utils.model_related as mr
from utils.log_writer import getLogWritter

# Esto es para tener el logger
logger = getLogWritter(__name__)

def load_model(args: dict, fine_tuning:bool = True, capas_entrenar_final = -1) -> torch.nn.Module:

    model = None
    model_name = args[cons.NAME_MODEL]
    model_weights_path = args[cons.NAME_MODEL_WEIGHTS]

    logger.debug(f"model_weights_path: {model_weights_path}")

    if model_weights_path is not None and os.path.isfile(model_weights_path):        
        model_weights_path = model_weights_path
        weights = None
    else: 
        weights = cons.DEFAULT_MODEL_WEIGHTS

    logger.info(f"Cargando los siguientes pesos: \'{weights if weights is not None else model_weights_path}\'")

    model = cons.SWITCH_MODELOS[model_name](weights = weights)

    if fine_tuning: 
        model = mr.fine_tuning(model=model, model_name=model_name, 
                    outputs=args[cons.NAME_NUMBER_CLASSES])
                
    if model_weights_path:
        # Es importante cargarlo *DESPUES* del fine tuning
        model.load_state_dict(torch.load(model_weights_path))        
    
    model = mr.transfer_learning(model, capas_entrenar_final)
        
    return model

def load_datasets(args:dict) -> dict:

    transform = transforms.Compose([
        # Aquí se puede añadir el aumento de datos, aunque prefiero que estén guardados en ficheros.
        transforms.Resize((cons.HEIGHT_IMAGES, cons.WIDTH_IMAGES)),
        transforms.ToTensor()
    ])

    return {
        folder_name:
        datasets.ImageFolder(root = os.path.join(args[cons.NAME_DATA_PATH], folder_name),
                             transform = transform) 
        for folder_name in cons.LIST_FOLDER_NAMES
    }

def load_samplers(datasets: Tuple)-> dict:

    return {        
        #folder_name: torch.utils.data.SequentialSampler(folder_name)
        folder_name: torch.utils.data.RandomSampler(datasets[folder_name])
        for folder_name in cons.LIST_FOLDER_NAMES
    }

# TODO: Por comprobar de que está bien
def load_data_loaders(args: dict, datasets:dict) -> dict:

    samplers = load_samplers(datasets=datasets)

    return {
        folder_name : torch.utils.data.DataLoader(
            datasets[folder_name], sampler=samplers[folder_name],
            batch_size=args[cons.NAME_BATCH_SIZE],
            # pin_memory=args.pin_mem,
            # drop_last=True,
            # shuffle = True # Con el Sampler debería hacerse automáticamente.
        )
        for folder_name in cons.LIST_FOLDER_NAMES
    }

def train_model(args: dict, model, samplers, device):
    
    logger.info( ('-' * cons.NUM_GUIONES) + "Starting to train the model" + ('-' * cons.NUM_GUIONES) )

    # Preparando las variables

    model = model.to(device)
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
    #for epoch in range(args[cons.NAME_EPOCS]):
    with logging_redirect_tqdm():
        for epoch in tqdm(range(args[cons.NAME_EPOCS]), desc="Epoch"):
        #logger.info(f"Epoch [{epoch + 1}/{args[cons.NAME_EPOCS]}]:")

            model.train(True)
            avg_loss = mr.train_one_epoch(model=model, epoch_index=epoch, 
                                        training_loader=samplers[cons.TRAIN_FOLDER_NAME], 
                                        loss_func=loss_fn, optimizer=optimizer)
            running_val_loss = 0.0

            model.eval()
                
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(samplers[cons.VALIDATION_FOLDER_NAME]):                                
                    vinputs, vlabels = vdata                
                    voutputs = model(vinputs)                
                    vloss = loss_fn(voutputs, vlabels)
                    running_val_loss += vloss

                    logger.debug("vinputs:\n", vinputs)
                    logger.debug("vlabels:\n", vlabels)
                    logger.debug("voutputs:\n", voutputs)

        
            avg_val_loss = running_val_loss / (i + 1)
            logger.info(f"Avg.loss: {avg_loss} | Avg.validation loss: {avg_val_loss}")                    

            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                model_name = f"model_{epoch}_{timestamp}.pth"
                model_path = os.path.join(args[cons.PARTIAL_MODELS_PATH], 
                                        model_name) 

                torch.save(model.state_dict(), model_path)

            # Para el early stopping
            if early_stopper.early_stop(running_val_loss):
                logger.info(f"Stopping the training. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")            
                break
    
    logger.info(('-' * cons.NUM_GUIONES) + " Training ended " + ('-' * cons.NUM_GUIONES))
    

def saving_the_model(args: dict, model):
    model_name = cons.OUTPUT_MODEL_NAME(name=args[cons.NAME_MODEL], number_clases=args[cons.NAME_NUMBER_CLASSES])
    logger.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)

def inference(args: dict, model, samplers):
    model.eval()

    
def get_training_time_HMS_format(time_start, time_end):
    intervalo = time_end - time_start

    days, seconds = intervalo.days, intervalo.seconds

    h = days * 24 + seconds // 3600
    m = (seconds % 3600) // 60
    s = (seconds % 60)

    return f"{h:02}:{m:02}:{s:02}"

def main(args: dict):

    # Estableciendo el dispositivo en el que se va a trabajar.    
    device = "cpu"
    if "cuda" in args[cons.NAME_DEVICE]:
        logger.debug(f"cuda in {args[cons.NAME_DEVICE]}")
        if not torch.cuda.is_available():
            logger.info(" 'torch.cuda' is not available ==> cpu")            
            #device = "cpu"
        else:
            device = args[cons.NAME_DEVICE]
    #else: device = "cpu"

    device = torch.device(device)
    logger.info(f"Device: {device}")

    datasets = load_datasets(args)
    
    logger.debug(
        "\n".join([f"len(dataset): {len(dataset)}\ndataset:\n{str(dataset)}" for dataset in datasets])
    )

    logger.debug(f"Datasets:\n{datasets}\n---")
    samplers = load_data_loaders(args, datasets)
    
    logger.debug(
        "\n".join([f"len(sampler): {len(sampler)}\nsampler:\n{str(sampler)}" for sampler in samplers])
    )

    
    logger.debug(f"Samplers:\n{samplers}\n---")

    model = load_model(args)
    logger.info(f"Model: {model}")
    #save_model = train_model(args, model=model, samplers=samplers, device=device)
    t_inicio = datetime.now()
    train_model(args, model=model, samplers=samplers, device=device)
    tiempo_entrenamiento = get_training_time_HMS_format(t_inicio, datetime.now())
    logger.info(f"Tiempo entrenamiento: {tiempo_entrenamiento}")
        
    mr.evaluate_model(model=model,dataloader=samplers[cons.VALIDATION_FOLDER_NAME], device=device)
    #accuracy, roc_auc, pr_auc, f1, conf_matrix = metricas
    saving_the_model(args, model)

# Esto lo hago así porque no se me ha ocurrido de otra forma
def poner_nivel_a_todos_logggers():
    lista_loggers = [ap.logger, mr.logger, logger]

    for _logger in lista_loggers:
        _logger.setLevel(cons.loggin_level)

if __name__ == "__main__":
    args = ap.get_dict_args()
    poner_nivel_a_todos_logggers()   
    main(args)