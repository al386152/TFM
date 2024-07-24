import os

import torch
from torch.distributed import barrier

from datetime import datetime

import utils.constants as cons
import utils.metrics as m
from utils.operations import GET_TIME_HMS_FORMAT, MEAN_TIMES, MULTIPLY_TIME
from utils.log_writer import getLogWritter
from utils.model_related import evaluate_model

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
            #if self.counter >= self.patience:
            #    return True
            return self.counter >= self.patience
        return False
# --  Fin class EarlyStopper -- #


def train_one_epoch(model, training_loader, device, estimacion_duracion,
                    loss_func=torch.nn.CrossEntropyLoss(), optimizer = None, is_main_device=True, debuging=False):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if optimizer is None else optimizer
    size_batches = len(training_loader)    
    log_info_cada = int(size_batches * cons.TANTO_POR_UNO_LOGS_PRINT) if int(size_batches * cons.TANTO_POR_UNO_LOGS_PRINT) > 0 else 1

    if is_main_device:
        tiempos = list()
        estimacion_fin, estimacion_fin_todos = estimacion_duracion
    
    for i, data in enumerate(training_loader):
        if is_main_device:
            tiempo_inicio = datetime.now()
            if debuging or (i % log_info_cada == 0):
                info = f"Batch: [{i + 1}/{size_batches}] - {estimacion_fin} || {estimacion_fin_todos}"
                logger.info(f"{info}")

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
            if debuging or (i % log_info_cada == 0):
                logger.info(f"loss: {loss}")
                logger.debug(f"tiempo_inicio: {str(tiempo_inicio)}")
            ahora = datetime.now()
            #tiempos.append(datetime.now() - tiempo_inicio)
            logger.debug(f"ahora: {str(ahora)}")
            tiempos.append(ahora - tiempo_inicio)
            #media_est_fin = sum(tiempos)/len(tiempos)
            media_est_fin = MEAN_TIMES(tiempos)
            logger.debug(f"media_est_fin: {str(media_est_fin)}")

            media_est_fin_todos = MULTIPLY_TIME(media_est_fin, (size_batches - i) )
            logger.debug(f"media_est_fin_todos: {str(media_est_fin_todos)}")
            estimacion_fin = GET_TIME_HMS_FORMAT(media_est_fin)
            estimacion_fin_todos = GET_TIME_HMS_FORMAT(media_est_fin_todos)
        
        if is_main_device:
            # No guardo la lista de los tiempos de una iteración a otra porque acabaría consumiéndose demasiada memoria (nº lotes * nº epochs) para un print.
            estimacion_duracion = (media_est_fin, MULTIPLY_TIME(media_est_fin, size_batches))

    # estimacion_duracion cambia de valor en el hilo principal, que es el que muestra los datos, mientras que en el resto de hilos se mantiene igual.
    return estimacion_duracion
# --  Fin train_one_epoch -- #

def train_model(args: dict, model, dataloaders, is_main_device, device, lista_metricas):
    
    if is_main_device:
        logger.info( ('-' * cons.NUM_GUIONES) + "Starting to train the model" + ('-' * cons.NUM_GUIONES) )

    # Preparando las variables
    best_metric = float('inf') # Número imposible para que en la primera iteración sea menor sí o sí.
    partial_models_path = args[cons.PARTIAL_MODELS_PATH] 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=args[cons.EARLY_STOPPING_PATIENCE] if args[cons.EARLY_STOPPING_PATIENCE] != -1 else args[cons.EPOCS], 
                                 min_delta=args[cons.EARLY_STOPPING_MIN_DELTA])
    optimizer = torch.optim.SGD(model.parameters(), lr=args[cons.LEARNING_RATE], momentum=0.9)
    
    # Creando carpetas para las salidas  
    if not os.path.isdir(partial_models_path) and is_main_device:
        os.mkdir(partial_models_path)

    # Si la carpeta donde se guardan las salidas no está creada y otro hilo trata de acceder ==> fallo.
    if args[cons.IS_DISTRIBUTED]:
        barrier()

    # Estas siguientes líneas es para tener una estimación del tiempo que se va a tardar a partir de las iteraciones anteriores.
    if is_main_device:
        tiempos = list()
        estimacion_fin = "(Tiempo estimado por época)"
        estimacion_fin_todos = "(Tiempo estimado total restante)"

    est_duracion_batch = ("(Tiempo estimado por lote)", "(Tiempo estimado total restante)") 

    num_epochs = args[cons.EPOCS]
    # Entrenando
    for epoch in range(num_epochs):       
        
        if is_main_device:
            tiempo_inicio = datetime.now()
            info = f"Epoch: [{epoch + 1}/{num_epochs}] - {estimacion_fin} || {estimacion_fin_todos}\n"
            logger.info(info)

        model.train(True)
        est_duracion_batch = train_one_epoch(model=model, device=device, 
                                                       training_loader=dataloaders[cons.TRAIN_FOLDER_NAME],  
                                                       loss_func=loss_fn, optimizer=optimizer, 
                                                       estimacion_duracion=est_duracion_batch, 
                                                       is_main_device=is_main_device, 
                                                       debuging=args[cons.SHOW_DEBUG_OUTPUTS])

        dict_resultados = evaluate_model(model=model, dataloader=dataloaders[cons.VALIDATION_FOLDER_NAME], device=device, 
                                         is_main_device=is_main_device, lista_metricas=lista_metricas, args=args, 
                                         loss_fn=loss_fn, save_confusion_matrix=False, nombre_prueba="Validation")

        if dict_resultados[cons.MAIN_METRIC] < best_metric:
            best_metric = dict_resultados[cons.MAIN_METRIC]
            if is_main_device :
                model_name = f"{args[cons.MODEL]}_{args[cons.NUMBER_CLASSES]}_{timestamp}.pth"
                model_path = os.path.join(args[cons.PARTIAL_MODELS_PATH], 
                                        model_name) 

                torch.save(model.state_dict(), model_path)

        # Para el early stopping
        if early_stopper.early_stop(dict_resultados[cons.MAIN_METRIC]):
            if is_main_device:
                logger.info(f"Stopping the training. {cons.MAIN_METRIC}: {dict_resultados[cons.MAIN_METRIC]}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")
            break

        if is_main_device:
            logger.debug(f"tiempo_inicio: {str(tiempo_inicio)}")            
            ahora = datetime.now()
            
            logger.debug(f"ahora: {str(ahora)}")            
            tiempos.append(ahora - tiempo_inicio)
            
            media_est_fin = MEAN_TIMES(tiempos)        
            logger.debug(f"media_est_fin: {str(media_est_fin)}")            

            media_est_fin_todos = MULTIPLY_TIME(media_est_fin, (num_epochs - epoch) )
            logger.debug(f"media_est_fin_todos: {str(media_est_fin_todos)}")
                         
            estimacion_fin = GET_TIME_HMS_FORMAT(media_est_fin)
            estimacion_fin_todos = GET_TIME_HMS_FORMAT(media_est_fin_todos)
    
    if is_main_device:
        logger.info(('-' * cons.NUM_GUIONES) + " Training ended " + ('-' * cons.NUM_GUIONES))

# -- Fin train_model -- #
