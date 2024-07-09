import torch
import numpy as np
#from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score, ConfusionMatrix

from datetime import datetime
from .common_operations import GET_TIME_HMS_FORMAT, MEAN_TIMES, MULTIPLY_TIME

from .log_writer import getLogWritter

logger = getLogWritter(__name__)

def transfer_learning(model:torch.nn.Module, capas_entrenar_final:int = -1, is_main_device = True):

    if capas_entrenar_final == -1:
        return model
    # else: se congelan las capas ==>
    if is_main_device:
        _info = f"s {capas_entrenar_final} últimas" if capas_entrenar_final > 1 else " última"
        logger.info(f"Preparando congelar todas las capas menos la {_info} para el transfer learning")
    
    capas_congelar = list(model.children())[: -1 * capas_entrenar_final] if capas_entrenar_final > 0 else model.children()

    for child in capas_congelar:
        if is_main_device:
            logger.debug(f"{str(child)}")
        for param in child.parameters():
            param.requires_grad = False   

    if is_main_device:
        logger.info(f"{'-'*4} Fin función transfer learning {'-'*4}")

    return model

# https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py
def fine_tuning(model, model_name, outputs, is_main_device):
    
    logger.info("Adding finne tuning layers")                        
    
    #if model_name == "vgg19":
    if "vgg" in model_name:
        classifier_layer = model.classifier[6]        
    #elif model_name == "resnet50":
    elif "resnet" in model_name:
        classifier_layer = model.fc
    elif "densenet" in model_name:
        classifier_layer = model.classifier
    else: 
        logger.warning(f"Han habido varias comprobaciones antes, ¿cómo has llegado aquí?. 'model_name: {model_name}'")
        classifier_layer = None

    num_ftrs = classifier_layer.in_features
    if is_main_device:
        logger.debug(f"num_ftrs: {str(num_ftrs)}")            

    model.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, outputs),
            #torch.nn.LayerNorm(outputs)
        )

    return model

def train_one_epoch(model, epoch_index, training_loader, loss_func=torch.nn.CrossEntropyLoss(), optimizer = None, is_main_device=True):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if optimizer is None else optimizer
    running_loss= 0
    last_loss = 0
    size_batches = len(training_loader)    
    
    if is_main_device:
        tiempos = list()
        estimacion_fin = "(Tiempo estimado por lote)"
        estimacion_fin_todos = "(Tiempo estimado total restante)"
    
    for i, data in enumerate(training_loader):
        if is_main_device:
            tiempo_inicio = datetime.now()
            info_print = f"Batch: [{i:02}/{size_batches}] - {estimacion_fin} || {estimacion_fin_todos}"
            logger.info(f"{info_print}")
        # Every data instance is an input + label pair
        inputs, labels = data        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            info_batch_loss = f"  batch {i + 1} loss: {last_loss}"
            tb_x = epoch_index * len(training_loader) + i + 1
            info_loss_train=f"Loss/train: {last_loss}/{tb_x}"
            running_loss = 0.
            if is_main_device:
                logger.info(f"Batch: [{i}/{size_batches}] {info_batch_loss}\n{info_loss_train} - {estimacion_fin} || {estimacion_fin_todos}")
        
        if is_main_device:
            logger.debug(f"tiempo_inicio: {str(tiempo_inicio)}")
            ahora = datetime.now()
            #tiempos.append(datetime.now() - tiempo_inicio)
            logger.debug(f"ahora: {str(ahora)}")
            tiempos.append(ahora - tiempo_inicio)
            #_estimacion_fin = sum(tiempos)/len(tiempos)
            _estimacion_fin = MEAN_TIMES(tiempos)
            logger.debug(f"_estimacion_fin: {str(_estimacion_fin)}")

            _estimacion_fin_todos = MULTIPLY_TIME(_estimacion_fin, (size_batches - i) )
            logger.debug(f"_estimacion_fin_todos: {str(_estimacion_fin_todos)}")
            estimacion_fin = GET_TIME_HMS_FORMAT(_estimacion_fin)
            estimacion_fin_todos = GET_TIME_HMS_FORMAT(_estimacion_fin_todos)

    return last_loss

def get_list_metrics(num_classes:int, task="multiclass"):
    return [ Accuracy(task = task, num_classes = num_classes), 
             AUROC(task = task, num_classes = num_classes), 
             AveragePrecision(task = task, num_classes = num_classes), 
             F1Score(task = task, num_classes = num_classes), 
             ConfusionMatrix(task = task, num_classes = num_classes)]

def update_metrics(list_metrics: list, outputs, labels):
    for metric in list_metrics:
        metric.update(outputs, labels)

def evaluate_model(model, dataloader, device, is_main_device, num_classes, task="multiclass"):
    model.eval()

    acc = Accuracy(task=task,num_classes=num_classes)
    auc = AUROC(task=task,num_classes=num_classes)
    avg_prec = AveragePrecision(task=task,num_classes=num_classes)
    f1_scr = F1Score(task=task,num_classes=num_classes)
    cnf_matrix = ConfusionMatrix(task=task,num_classes=num_classes)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            #all_preds.append(preds.cpu().numpy())
            #all_labels.append(labels.cpu().numpy())
            acc.update(outputs, labels)
            auc.update(outputs, labels)
            avg_prec.update(outputs, labels)
            f1_scr.update(outputs, labels)
            cnf_matrix.update(outputs, labels)
    
    #all_preds = np.concatenate(all_preds)
    #all_labels = np.concatenate(all_labels)
    
    accuracy = acc.compute().item()
    roc_auc = auc.compute().item()
    pr_auc = avg_prec.compute()
    f1 = f1_scr.compute()
    conf_matrix = cnf_matrix.compute()
    
    if is_main_device:
        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'ROC-AUC: {roc_auc:.4f}')
        logger.info(f'PR-AUC: {pr_auc:.4f}')
        logger.info(f'F1 Score: {f1:.4f}')
        logger.info(f'Confusion Matrix:\n{conf_matrix}')

    return accuracy, roc_auc, pr_auc, f1, conf_matrix

# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False