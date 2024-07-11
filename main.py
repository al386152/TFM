import os

from datetime import datetime

import torch.utils
import torch

from torch import distributed, cuda

import utils.arguments_parser as ap
import utils.constants as cons
from utils.operations import GET_TIME_HMS_FORMAT

import utils.training as t
import utils.model_related as mr
from utils.log_writer import getLogWritter, set_level
from utils.metrics import  get_list_metrics, logger as metrics_logger
from utils.data_loaders import load_datasets, load_data_loaders, logger as dl_logger

# Esto es para tener el logger
logger = getLogWritter(__name__)

# TODO: Revisar esto (seguro que hay una forma mucho mejor de hacerlo) y mirar de moverlo a otra carpeta.
def setup_gpu(args:dict):

    # Estableciendo el dispositivo en el que se va a trabajar.
    device = "cpu"
    worldsize = 1
    if "cuda" in args[cons.DEVICE]:
        logger.debug(f"cuda in {str(args[cons.DEVICE])}")
        if not cuda.is_available():
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
        device = int(os.environ["LOCAL_RANK"])
        worldsize = int(os.environ["WORLD_SIZE"])                
    else:
        device = torch.device(device)

    return device, worldsize
# -- Fin setup_gpu -- #

def main(args: dict):    

    device, _ = setup_gpu(args=args)

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device:
        logger.info(f"Device: {device}")

    model = mr.load_model(args=args, device=device, is_main_device=is_main_device)
    if is_main_device:
        logger.info(f"Model: {model}")

    datasets = load_datasets(args, is_main_device)
    if is_main_device:
        logger.debug("\n".join([f"len(dataset): {len(dataset)}\ndataset:\n{dataset}" for dataset in datasets]))

    data_loaders = load_data_loaders(args, datasets)
    if is_main_device:
        logger.debug("\n".join([f"len(dataloader): {len(dataloader)}\dataloader:\n{dataloader}" for dataloader in data_loaders]))

    metricas = get_list_metrics(args[cons.NUMBER_CLASSES])

    if is_main_device:
        t_inicio = datetime.now()
    t.train_model(args, model=model, dataloaders=data_loaders, is_main_device=is_main_device, device=device, lista_metricas=metricas)
    if is_main_device:
        tiempo_entrenamiento = GET_TIME_HMS_FORMAT((datetime.now() - t_inicio))
        logger.info(f"Tiempo entrenamiento: {tiempo_entrenamiento}")            
    
    if is_main_device:
        mr.saving_the_model(args, model)
        mr.evaluate_model(model=model,dataloader=data_loaders[cons.VALIDATION_FOLDER_NAME], 
                          device=device, is_main_device=is_main_device, lista_metricas=metricas)
# -- Fin main -- #

# Esto lo hago así porque no se me ha ocurrido de otra forma de poner el nivel correcto a todos los loggers de todos los scripts
def poner_nivel_a_todos_loggers():
    lista_loggers = [ap.logger, dl_logger, metrics_logger, mr.logger, t.logger, logger]

    #for _logger in lista_loggers:
        #_logger.setLevel(cons.loggin_level)
    set_level(lista_loggers, cons.loggin_level)
# -- Fin poner_nivel_a_todos_loggers -- #

if __name__ == "__main__":
    args = ap.get_dict_args()
    poner_nivel_a_todos_loggers()   
    main(args)
