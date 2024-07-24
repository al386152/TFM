import os
import sys
import logging

from datetime import datetime
from typing import List

from utils.operations import GET_FECHA_INICIO_EJECUCION
from . import constants as cons

def prepare_log_file() -> str:

    the_date = GET_FECHA_INICIO_EJECUCION()
    log_path_with_date = os.path.join(cons.DEFAULT_LOG_FOLDER, str(the_date.date()))

    if not os.path.isdir(log_path_with_date):
        os.mkdir(log_path_with_date)

    log_name = f"{str(the_date.time()).replace(':', '-')}_{cons.BASE_LOG_FILE_NAME}.log"

    return log_path_with_date, log_name

def set_level(loggers: List[logging.Logger], level):
    for logger in loggers:
        logger.setLevel(level)        
        for handler in logger.handlers:
            handler.setLevel(level)

# TODO: Finalizar esto.
def add_file_handler(loggers: List[logging.Logger]):
    
    log_path_with_date, log_name = prepare_log_file()

    file = os.path.join(log_path_with_date, log_name)

    for logger in loggers:                  
        fh = logging.FileHandler(file)
        # Para que se guarde con el mismo formato que como sale por pantalla
        fh.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(fh)    
            

def getLogWritter(name, logging_level=cons.loggin_level, 
                copiar_a_stdout = True, logger = None)->logging.Logger:   

    # Realmente, solo el proceso 0 hace los logs 

    # ("LOCAL_RANK" not in os.environ) es true si se trabaja sin concurrencia
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        # TODO: mirar si mover esto a alguna otra parte.

        log_path_with_date, log_name = prepare_log_file()

        # Si el proceso 0 no ha creado la carpeta donde se van a guardar los logs, todos tienen que esperarse.     
        #barrier()        
                
        logging_level = logging._nameToLevel[logging_level]
        
        logging.basicConfig(filename=os.path.join(log_path_with_date, log_name), 
                            level=logging_level, format=cons.LOG_OUTPUT_FORMAT)

        logger = logging.getLogger(name)

        if copiar_a_stdout:
            # Redirigiendo la salida a la salida estándar
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging_level)        
            handler.setFormatter(logging.Formatter(cons.LOG_OUTPUT_FORMAT))
            logger.addHandler(handler)
        
        return logger
    return None
