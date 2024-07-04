import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

from datetime import datetime
from .common_operations import GET_TIME_HMS_FORMAT, MEAN_TIMES, MULTIPLY_TIME

from .log_writer import getLogWritter
from .constants import NUM_GUIONES

logger = getLogWritter(__name__)

def transfer_learning(model, capas_entrenar_final = -1):

    if capas_entrenar_final == -1:
        return model
    # else: congela las capas

    logger.info(f"Preparando congelar todas las capas menos las {capas_entrenar_final} últimas para el transfer learning")    
    # Esto es para congelar las capas
    list_model_children = list()
    
    logger.debug("for block in model.blocks:")
    for block in model.blocks:
        print(block)
        for child in block.children():
            logger.debug(f"{child} |",end=' ')    
            list_model_children.append(child)
            logger.debug("len(list(child.parameters())): ", len(list(child.parameters())))            
    logger.debug('-' * NUM_GUIONES)
    
    logger.debug("len(list_model_children): ", len(list_model_children))
    if capas_entrenar_final != 0:
        list_model_children = list_model_children[: -1 * capas_entrenar_final]
    # else: list_model_children = list_model_children
    logger.debug("len(list_model_children): ", len(list_model_children))    

    for child in list_model_children:
        logger.debug(f"{child} |",end=' ')
        for param in child.parameters():
            #print(f"{param} |\|",end=' ')
            param.requires_grad = False   

    # Nota: En algún punto, ya se añade una capa con las salidas esperadas.
    logger.info(f"{'-'*4} Fin función transfer learning {'-'*4}")

    return model

def fine_tuning(model, model_name, outputs, ):
    
    logger.info("Adding finne tuning layers")                        

    #if model_name == "vgg19":
    if "vgg" in model_name:
        num_ftrs = model.classifier[6].in_features
        logger.debug("model.classifier[6].in_features: ", model.classifier[6].out_features)            

        model.classifier[6] = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, outputs),
                torch.nn.LayerNorm(outputs)
            )
    #elif model_name == "resnet50":
    elif "resnet" in model_name:
        
        num_ftrs = model.fc.in_features
        logger.debug("model.fc.in_features: ", model.fc.in_features)

        model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, outputs),
                torch.nn.LayerNorm(outputs)
            )
    #else: otros casos 


    return model

def train_one_epoch(model, epoch_index, training_loader, loss_func=torch.nn.CrossEntropyLoss(), optimizer = None):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if optimizer is None else optimizer
    running_loss= 0
    last_loss = 0
    size_batches = len(training_loader)    
    
    tiempos = list()
    estimacion_fin = "(Tiempo estimado por lote)"
    estimacion_fin_todos = "(Tiempo estimado total)"
    
    for i, data in enumerate(training_loader):
        tiempo_inicio = datetime.now()
        info_print = f"Batch: [{i}/{size_batches}] - {estimacion_fin} || {estimacion_fin_todos}"
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
            logger.info(f"Batch: [{i}/{size_batches}] {info_batch_loss}\n{info_loss_train} - {estimacion_fin} || {estimacion_fin_todos}")
        
        logger.debug("tiempo_inicio: ", tiempo_inicio)
        ahora = datetime.now()
        #tiempos.append(datetime.now() - tiempo_inicio)
        logger.debug("ahora: ", ahora)
        tiempos.append(ahora - tiempo_inicio)
        #_estimacion_fin = sum(tiempos)/len(tiempos)
        _estimacion_fin = MEAN_TIMES(tiempos)
        logger.debug("_estimacion_fin: ", _estimacion_fin)

        _estimacion_fin_todos = MULTIPLY_TIME(_estimacion_fin, size_batches)
        logger.debug("_estimacion_fin_todos: ", _estimacion_fin_todos)
        estimacion_fin = GET_TIME_HMS_FORMAT(_estimacion_fin)
        estimacion_fin_todos = GET_TIME_HMS_FORMAT(_estimacion_fin_todos)

    return last_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    #roc_auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovo') if len(np.unique(all_labels)) > 2 else roc_auc_score(all_labels, all_preds)
    roc_auc = 0.0
    pr_auc = average_precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
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