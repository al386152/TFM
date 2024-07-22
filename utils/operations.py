import os
from datetime import datetime, timedelta
from torch import distributed, cuda
import torch.utils

import utils.constants as cons

fecha = str(datetime.now().replace(microsecond=0)).replace(':', '-').replace(' ', '_')

def OUTPUT_MODEL_NAME(name:str, number_clases:int)->str:
    return f"model_{name}_{number_clases}_outputs.pth"
# -- Fin OUTPUT_MODEL_NAME -- #

# Se asume que la lista no está vacía
def GET_TIME_HMS_FORMAT(the_time, time_format="%H:%M:%S.%f")->str:
    
    # Si es timedelta, lo volvemos a transformar en un datetime y, de ahí, a un string con el formato deseado.
    if isinstance(the_time, timedelta):
        # Se ha dado el caso de que si justo ha ocurrido en el mismo microsegundo, timedelta elimina los microsegundos de su string
        if the_time.microseconds == 0:
            _time_format = time_format[: time_format.index(".%f") ]
        else:
            _time_format = time_format
        the_time = datetime.strptime(str(the_time), _time_format)
    
    return the_time.strftime(time_format)
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
    return os.path.join(cons.IMAGES_FOLDER_NAME, fecha)

def GET_FECHA_INICIO_EJECUCION():
    return fecha