import os

import torch

from datetime import datetime
import utils.constants as cons
import utils.metrics as m
from .operations import GET_TIME_HMS_FORMAT, MEAN_TIMES, MULTIPLY_TIME

from .log_writer import getLogWritter

logger = getLogWritter(__name__)

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
# --  Fin class EarlyStopper -- #


def train_one_epoch(model, training_loader, device, loss_func=torch.nn.CrossEntropyLoss(), optimizer = None, is_main_device=True):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if optimizer is None else optimizer
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

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_main_device:
            logger.debug(f"Inputs shape: {inputs.shape}, min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.mean()}")
            logger.debug(f"Labels shape: {labels.shape}, labels: {labels}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()            
        
        if is_main_device:
            logger.info(f"loss: {loss}")
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
# --  Fin train_one_epoch -- #

def train_model(args: dict, model, dataloaders, is_main_device, device, lista_metricas):
    
    if is_main_device:
        logger.info( ('-' * cons.NUM_GUIONES) + "Starting to train the model" + ('-' * cons.NUM_GUIONES) )

    # Preparando las variables
    best_loss = float('inf') # Número imposible para que en la primera iteración sea menor sí o sí.
    partial_models_path = args[cons.PARTIAL_MODELS_PATH] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=3, min_delta=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Creando carpetas para las salidas  
    if not os.path.isdir(partial_models_path):
        os.makedirs(partial_models_path)

    # Entrenando
    if is_main_device:
        tiempos = list()
        estimacion_fin = "(Tiempo estimado por época)"
        estimacion_fin_todos = "(Tiempo estimado total restante)"

    num_epochs = args[cons.EPOCS]

    for epoch in range(num_epochs):       
        
        if is_main_device:
            tiempo_inicio = datetime.now()
            info_print = f"Epoch: [{epoch}/{num_epochs}] - {estimacion_fin} || {estimacion_fin_todos}"
            logger.info(info_print)

        model.train(True)
        avg_loss = train_one_epoch(model=model, device=device,
                                    training_loader=dataloaders[cons.TRAIN_FOLDER_NAME], 
                                    loss_func=loss_fn, optimizer=optimizer, is_main_device=is_main_device)
        running_val_loss = 0.0

        model.eval()
            
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, data in enumerate(dataloaders[cons.VALIDATION_FOLDER_NAME]):                                
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                if is_main_device:
                    logger.debug(f"Inputs shape: {inputs.shape}, min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.mean()}")
                    logger.debug(f"Labels shape: {labels.shape}, labels: {labels}")

                outputs = model(inputs)                

                if is_main_device:
                    logger.debug(f"Validation Outputs shape: {outputs.shape}, Validation Labels shape: {labels.shape}")

                loss = loss_fn(outputs, labels)
                running_val_loss += loss
                
                if is_main_device:
                    logger.debug(f"inputs:\n{str(inputs)}" )
                    logger.debug(f"labels:\n{str(labels)}" )
                    logger.debug(f"outputs:\n{str(outputs)}" )
                
                m.update_metrics(lista_metricas, outputs=outputs, labels=labels)

        lista_resultados = m.get_metrics(lista_metricas)
        m.reset_list_metrics(lista_metricas)

        avg_val_loss = running_val_loss / (i + 1)
        if is_main_device:
            logger.info(f"Avg.loss: {avg_loss} | Avg.validation loss: {avg_val_loss}")                    

        if is_main_device:
            logger.debug(f"Resultados: {lista_resultados}")
            for name, metric in lista_resultados:
                logger.info(f'{name}: {metric:.4f}' if name != "ConfusionMatrix" else f"{name}:\n{metric}")

        if is_main_device and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_name = f"model_{epoch}_{timestamp}.pth"
            model_path = os.path.join(args[cons.PARTIAL_MODELS_PATH], 
                                    model_name) 

            torch.save(model.state_dict(), model_path)

        # Para el early stopping
        if early_stopper.early_stop(running_val_loss):
            logger.info(f"Stopping the training. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")            
            break
        
        if is_main_device:
            logger.debug(f"tiempo_inicio: {str(tiempo_inicio)}")
            ahora = datetime.now()
            #tiempos.append(datetime.now() - tiempo_inicio)
            logger.debug(f"ahora: {str(ahora)}")
            tiempos.append(ahora - tiempo_inicio)
            #_estimacion_fin = sum(tiempos)/len(tiempos)
            _estimacion_fin = MEAN_TIMES(tiempos)
            logger.debug(f"_estimacion_fin: {str(_estimacion_fin)}")

            _estimacion_fin_todos = MULTIPLY_TIME(_estimacion_fin, (num_epochs - i) )
            logger.debug(f"_estimacion_fin_todos: {str(_estimacion_fin_todos)}")
            estimacion_fin = GET_TIME_HMS_FORMAT(_estimacion_fin)
            estimacion_fin_todos = GET_TIME_HMS_FORMAT(_estimacion_fin_todos)
    
    if is_main_device:
        logger.info(('-' * cons.NUM_GUIONES) + " Training ended " + ('-' * cons.NUM_GUIONES))

# -- Fin train_model -- #

