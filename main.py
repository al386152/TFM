import os

from datetime import datetime
from torch.nn.parallel import DistributedDataParallel

import utils.arguments_parser as ap
import utils.constants as cons
from utils.operations import GET_TIME_HMS_FORMAT

import utils.training as t
import utils.model_related as mr
from utils.log_writer import getLogWritter, set_level
from utils.metrics import get_list_metrics, logger as metrics_logger
from utils.data_loaders import load_datasets, load_data_loaders, logger as dl_logger
from utils.operations import setup_gpu
from utils.optuna_related import main_optuna

# Esto es para tener el logger
logger = getLogWritter(__name__)

def main(args: dict):    

    device, _ = setup_gpu(args=args, logger=logger)

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device:
        logger.info(f"Device: {device}")

    model = mr.load_model(args=args, device=device, is_main_device=is_main_device)
    model = model.to(device)
    if args[cons.IS_DISTRIBUTED]:
        model = DistributedDataParallel(model, device_ids=[device])        

    if is_main_device:
        logger.info(f"Model: {model}")

    datasets = load_datasets(args, is_main_device)
    if is_main_device:
        logger.debug("\n".join([f"len(dataset): {len(dataset)}\ndataset:\n{dataset}" for dataset in datasets]))

    data_loaders = load_data_loaders(args, datasets)
    if is_main_device:
        logger.debug("\n".join([f"len(dataloader): {len(dataloader)}\dataloader:\n{dataloader}" for dataloader in data_loaders]))

    metricas = get_list_metrics(args[cons.NUMBER_CLASSES], device=device)

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

if __name__ == "__main__":
    args = ap.get_dict_args()
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
    # Esto lo hago así porque no se me ha ocurrido de otra forma de poner el nivel correcto a todos los loggers de todos los scripts
        lista_loggers = [ap.logger, dl_logger, metrics_logger, mr.logger, t.logger, logger]
        set_level(lista_loggers, cons.loggin_level)
    
    if args[cons.IS_HYPERTUNING]:
        main_optuna(args)
    else:
        main(args)
# -- Fin verdadero main -- #
