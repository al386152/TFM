import os
from datetime import datetime, timedelta
from torch import distributed, cuda
import torch.distributed
import torch.utils
from collections import Counter

import utils.constants as cons

fecha = datetime.now().replace(microsecond=0)

def OUTPUT_MODEL_NAME(name:str, number_clases:int)->str:
    return f"model_{name}_{number_clases}_outputs.pth"
# -- Fin OUTPUT_MODEL_NAME -- #

# Se asume que la lista no está vacía
def GET_TIME_HMS_FORMAT(the_time, time_format="%H:%M:%S.%f")->str:
    
    # Si es timedelta, lo volvemos a transformar en un datetime y, de ahí, a un string con el formato deseado.
    if isinstance(the_time, timedelta):

        # Si por un casual el entrenamiento dura más de 1 día hay que hacer pasos especiales
        if the_time.days > 0:

            seconds = the_time.total_seconds()
            mseconds = the_time.microseconds

            h, resto = divmod(seconds, 3600)
            m, s = divmod(resto, 60)
            
            output = f"{int(h):02}:{int(m):02}:{int(s):02}.{mseconds}"
        else:
            # Se ha dado el caso de que si justo ha ocurrido en el mismo microsegundo, timedelta elimina los microsegundos de su string
            if the_time.microseconds == 0:
                _time_format = time_format[: time_format.index(".%f") ]    
            else:
                _time_format = time_format
            the_time = datetime.strptime(str(the_time), _time_format)
            output = the_time.strftime(time_format)
    
    return output
# -- Fin GET_TIME_HMS_FORMAT -- #

# Se asume que la lista no está vacía
def MEAN_TIMES(list_times: list)->timedelta:
    return timedelta(seconds=sum(map(lambda t: t.total_seconds(), list_times)) / len(list_times))
# -- Fin MEAN_TIMES -- #

def MULTIPLY_TIME(the_time, alpha:float)->timedelta:
    if isinstance(the_time, timedelta):
        return the_time * alpha
    else:
        return timedelta(seconds=datetime.timestamp(the_time) * alpha)
# -- Fin MULTIPLY_TIME -- #

def validate_dataset(dataset, logger):
    # Esta función se tiene que hacer solo si es el hilo principal.
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    for class_idx, count in class_counts.items():
        logger.info(f"Class {dataset.classes[class_idx]} ({class_idx}): {count} samples")
# -- Fin validate_dataset -- #

# TODO: Revisar esto (seguro que hay una forma mucho mejor de hacerlo)
def setup_gpu(args:dict, logger):

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


def GET_IMAGES_FOLDER_PATH():
    return os.path.join(cons.IMAGES_FOLDER_NAME, GET_FECHA_INICIO_EJECUCION_CON_FORMATO())
# -- Fin GET_IMAGES_FOLDER_PATH -- #

def GET_FECHA_INICIO_EJECUCION_CON_FORMATO():    
    return str(fecha.replace(microsecond=0)).replace(':', '-').replace(' ', '_')
# -- Fin GET_FECHA_INICIO_EJECUCION_CON_FORMATO -- #

def GET_FECHA_INICIO_EJECUCION():
    return fecha
# -- Fin GET_FECHA_INICIO_EJECUCION -- #

# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_spawn.py
def cleanup():    
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
# -- Fin cleanup -- # 

def get_proporcion_datasets(datasets:dict) -> dict:
    proporcion = {}
    for clave in cons.LIST_FOLDER_NAMES:            
        _proporcion = datasets[clave].get_proporcion_clase()        
                
        for elem in _proporcion:
            if elem not in proporcion:
                proporcion[elem] = 0    
            proporcion[elem] += _proporcion[elem]

    num_folders = len(cons.LIST_FOLDER_NAMES)
    return {elem: proporcion[elem]/num_folders for elem in proporcion}
# -- Fin get_proporcion_datasets -- #
