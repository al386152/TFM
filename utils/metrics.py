import os

from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, ConfusionMatrix
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import matplotlib.pyplot as plt

import utils.constants as cons
from utils.operations import GET_IMAGES_FOLDER_PATH
from utils.log_writer import getLogWritter

logger = getLogWritter(__name__)

def get_list_metrics(num_classes:int, device, task = "multiclass"):
    lista_metricas = [ 
            (cons.ACCURACY, Accuracy(task = task, num_classes = num_classes)),
            (cons.AUROC, AUROC(task = task, num_classes = num_classes)),
            (cons.AVERAGE_PRECISION, AveragePrecision(task = task, num_classes = num_classes)),
            (cons.F_ONE_SCORE, F1Score(task = task, num_classes = num_classes)),
            (cons.CONFUSION_MATRIX, ConfusionMatrix(task = task, num_classes = num_classes)),
            (cons.NORM_CONFUSION_MATRIX, ConfusionMatrix(task = task, num_classes = num_classes, normalize="true"))]
    
    # Moviéndo las métricas al dispositivo que toca.
    for _, metrica in lista_metricas:
        metrica.to(device)

    return lista_metricas
# -- Fin get_list_metrics -- #

def get_regression_list_metrics(num_outputs:int, device):
    lista_metricas = [ 
            # Este método va inmensamente lento, tanto que me salta el watchdog.
            # TODO: Mirar de, si eso, hacerlo en la última iteración
            #(cons.MAE, MeanAbsoluteError()),
            #(cons.MSE, MeanSquaredError(num_outputs = num_outputs)),
            #(cons.R2SCORE, R2Score(num_outputs = num_outputs)),
            ]
    
    # Moviéndo las métricas al dispositivo que toca.
    for _, metrica in lista_metricas:
        metrica.to(device)

    return lista_metricas
# -- Fin get_list_metrics -- #

def update_metrics(list_metrics: list, outputs, labels):
    for _, metric in list_metrics:
        metric.update(outputs, labels)
# -- Fin update_metrics -- #        

def get_metrics(list_metrics: list, args:dict, save_confusion_matrix = False, is_main_device=False):

    resultados = dict()
    for i in range(len(list_metrics)):    
        name, metric = list_metrics[i]
        if is_main_device:
            logger.debug(f"get_metrics - name: {name}, metric: {metric}")

        if name == cons.CONFUSION_MATRIX or name == cons.NORM_CONFUSION_MATRIX:
            
            if is_main_device:
                logger.debug(f"{name}")
        
            resultados[name] = metric.compute()

            if save_confusion_matrix:
                
                if is_main_device: 
                    logger.debug(f"Saving confusion matrix")
                
                metric.plot(cmap=plt.cm.Blues)
                path_images_folder = GET_IMAGES_FOLDER_PATH()

                if is_main_device:
                    if not os.path.isdir(path_images_folder):
                        os.makedirs(path_images_folder)
                        #os.mkdir(path_images_folder)
                        
                    name_file = f"{args[cons.MODEL]}_{name}{cons.IMAGES_FILE_FORMAT}"                    
                    name_file = os.path.join(path_images_folder, name_file)

                    plt.savefig(name_file, dpi=600, bbox_inches ='tight')
        elif name == cons.MAE or name == cons.MSE or name == cons.R2SCORE:
            resultados[name] = metric.compute()
            if is_main_device:
                logger.debug(f"resultados[{name}]: {resultados[name]}")
        else:            
            if is_main_device:
                logger.debug(f"Other metric")

            resultados[name] = metric.compute().item()
    #logger.info(f"Resultados: {resultados}")
    return resultados
# -- Fin get_metrics -- #
    
def reset_list_metrics(list_metrics: list):
    for _, metric in list_metrics:
        metric.reset()
# -- Fin reset_list_metrics -- #
