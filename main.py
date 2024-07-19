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
from utils.metrics import  get_list_metrics, get_list_metrics_with_CM, logger as metrics_logger
from utils.data_loaders import load_datasets, load_data_loaders, logger as dl_logger

# Esto es para tener el logger
logger = getLogWritter(__name__)

# TODO: Revisar esto (seguro que hay una forma mucho mejor de hacerlo)
def setup_gpu(args:dict):

    # Estableciendo el dispositivo en el que se va a trabajar.
    device = "cpu"
    worldsize = 1    
    if "cuda" in args[cons.DEVICE]:
        # ("LOCAL_RANK" not in os.environ) es true si se trabaja sin concurrencia
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.debug(f"cuda in {str(args[cons.DEVICE])}")
        if not cuda.is_available():
            if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
                logger.info("'torch.cuda' is not available ==> cpu")            
            #device = "cpu"
        else:
            if args[cons.IS_DISTRIBUTED]:
                if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
                    logger.info("Setting up distributed gpu") # TODO: pensar en un mensaje mejor para el log
                distributed.init_process_group(backend=cons.BACKEND)
                #device = int(os.environ["LOCAL_RANK"])
                #worldsize = int(os.environ["WORLD_SIZE"])                
            else:
                if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
                    logger.info("Setting one gpu") # TODO: pensar en un mensaje mejor para el log
                device = args[cons.DEVICE]
    #else: device = "cpu"
    
    if args[cons.IS_DISTRIBUTED]:
        device = int(os.environ["LOCAL_RANK"])
        worldsize = int(os.environ["WORLD_SIZE"])    
        cuda.set_device(device)            
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

    #metricas = get_list_metrics(args[cons.NUMBER_CLASSES], device=device)
    metricas = get_list_metrics_with_CM(args[cons.NUMBER_CLASSES], device=device)

    if is_main_device:
        t_inicio = datetime.now()
    t.train_model(args, model=model, dataloaders=data_loaders, is_main_device=is_main_device, device=device, lista_metricas=metricas)
    if is_main_device:
        tiempo_entrenamiento = GET_TIME_HMS_FORMAT((datetime.now() - t_inicio))
        logger.info(f"Tiempo entrenamiento: {tiempo_entrenamiento}")

    # Evaluando el modelo con el conjunto de "test"    
    if is_main_device:
        logger.info(f"{'-' * cons.NUM_GUIONES} Probando el modelo {'-' * cons.NUM_GUIONES}")
    
    mr.evaluate_model(model=model, dataloader=data_loaders[cons.TEST_FOLDER_NAME], device=device, 
                      is_main_device=is_main_device, lista_metricas=metricas, args=args)
    if is_main_device:
        logger.info(f"{'-' * cons.NUM_GUIONES} Fin test {'-' * cons.NUM_GUIONES}")

    if is_main_device:
        mr.saving_the_model(args, model)
        logger.info(f"{'-' * cons.NUM_GUIONES} Programa finalizado {'-' * cons.NUM_GUIONES}")
# -- Fin main -- #

# Esto lo hago así porque no se me ha ocurrido de otra forma de poner el nivel correcto a todos los loggers de todos los scripts
def poner_nivel_a_todos_loggers():
    lista_loggers = [ap.logger, dl_logger, metrics_logger, mr.logger, t.logger, logger]
    set_level(lista_loggers, cons.loggin_level)
# -- Fin poner_nivel_a_todos_loggers -- #

if __name__ == "__main__":
    args = ap.get_dict_args()
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        poner_nivel_a_todos_loggers()   
    main(args)
# -- Fin verdadero main -- #
