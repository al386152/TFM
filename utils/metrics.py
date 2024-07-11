from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, ConfusionMatrix

from .log_writer import getLogWritter

logger = getLogWritter(__name__)

def _mover_a_dispositivo(lista_metricas, device):

    for metrica in lista_metricas:
        metrica.to(device)

def get_list_metrics(num_classes:int, device, task="multiclass"):
    lista_metricas = [ 
             ("Accuracy", Accuracy(task = task, num_classes = num_classes)),
             ("AUROC", AUROC(task = task, num_classes = num_classes)),
             ("AveragePrecision", AveragePrecision(task = task, num_classes = num_classes)),
             ("F1Score", F1Score(task = task, num_classes = num_classes))]
    
    _mover_a_dispositivo(lista_metricas, device)

    return lista_metricas
# -- Fin get_list_metrics -- #

def get_list_metrics_with_CM(num_classes:int, device, task="multiclass"):
    lista_metricas = [ 
            ("Accuracy", Accuracy(task = task, num_classes = num_classes)),
            ("AUROC", AUROC(task = task, num_classes = num_classes)),
            ("AveragePrecision", AveragePrecision(task = task, num_classes = num_classes)),
            ("F1Score", F1Score(task = task, num_classes = num_classes)),
            ("ConfusionMatrix", ConfusionMatrix(task = task, num_classes = num_classes))]
    
    _mover_a_dispositivo(lista_metricas, device)

    return lista_metricas
# -- Fin get_list_metrics_with_CM -- #

def update_metrics(list_metrics: list, outputs, labels):
    for _, metric in list_metrics:
        metric.update(outputs, labels)
# -- Fin update_metrics -- #        

def get_metrics(list_metrics: list):
    return [metric.compute().item() for _, metric in list_metrics]
# -- Fin get_metrics -- #
    
def reset_list_metrics(list_metrics: list):
    for _, metric in list_metrics:
        metric.reset()
# -- Fin reset_list_metrics -- #