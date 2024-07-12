import os
import sys
import logging
from datetime import datetime
from typing import List

from . import constants as cons

def set_level(loggers: List[logging.Logger], level, copiar_a_stdout=True):
    for logger in loggers:
        logger.setLevel(level)        
        for handler in logger.handlers:
            handler.setLevel(level)
            
def getLogWritter(name, log_path = cons.DEFAULT_LOG_FOLDER, 
                log_name = cons.BASE_LOG_FILE_NAME, 
                logging_level=cons.loggin_level, 
                copiar_a_stdout = True)->logging.Logger:   

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    
    the_date = datetime.now()
    log_path = os.path.join(log_path, str(the_date.date()))
    log_name = f"{str(the_date.time().replace(microsecond=0)).replace(':', '')}_{log_name}.log"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    
    logging_level = logging._nameToLevel[logging_level]

    logging.basicConfig(filename=os.path.join(log_path, log_name), 
                        level=logging_level, format=cons.LOG_OUTPUT_FORMAT)

    logger = logging.getLogger(name)

    if copiar_a_stdout:
        # Redirigiendo la salida a la salida estándar
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging_level)        
        handler.setFormatter(logging.Formatter(cons.LOG_OUTPUT_FORMAT))
        logger.addHandler(handler)
    
    return logger
