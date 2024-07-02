import logging
import os
import sys
import constants as cons
from datetime import datetime

def getLogWritter(name, log_path = cons.DEFAULT_LOG_FOLDER, 
                log_name = cons.BASE_LOG_FILE_NAME, 
                logging_level=cons.loggin_level, 
                copiar_a_stdout = True)->logging.Logger:   

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    
    log_path = os.path.join(log_path, str(datetime.now().date()))

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    
    logging_level = logging._nameToLevel[logging_level]
    logging.basicConfig(filename=os.join(log_path, log_name), 
                        level=logging_level)
    
    logger = logging.getLogger(name, format=cons.LOG_OUTPUT_FORMAT)

    if copiar_a_stdout:
        # Redirigiendo la salida a la salida estándar
        handler = logging.StreamHandler(sys.out)
        handler.setLevel(logging_level)
        formatter = logging.Formatter(cons.LOG_OUTPUT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
    

