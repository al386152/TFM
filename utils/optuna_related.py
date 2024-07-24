import optuna

import os
import matplotlib.pyplot as plt
import torch

import utils.constants as cons
from utils.operations import GET_IMAGES_FOLDER_PATH
from utils.log_writer import getLogWritter, set_level, add_file_handler

from utils.training import train_one_epoch
import utils.model_related as mr
from utils.metrics import  get_list_metrics
from utils.data_loaders import load_datasets, load_data_loaders
from utils.operations import setup_gpu

logger = getLogWritter(__name__)
ARGS = None


def save_plot(args:dict, name:str):
    
    path_images_folder = GET_IMAGES_FOLDER_PATH()

    if not os.path.isdir(path_images_folder):
        #os.mkdir(path_images_folder)
        os.makedirs(path_images_folder)
            
    name_file = f"{name}{cons.IMAGES_FILE_FORMAT}"
    name_file = os.path.join(path_images_folder, name_file)

    plt.savefig(name_file, dpi=600, bbox_inches ='tight')
# -- FIN save_plot -- #

# Se asume que es el hilo principal (el "0")
def show_and_save_results(study:optuna.Study, args:dict):
    
    # Plot history
    optuna.visualization.matplotlib.plot_optimization_history(study)
    save_plot(args=args, name="Optimization_History")
    optuna.visualization.matplotlib.plot_intermediate_values(study)
    save_plot(args=args, name="Intermediate_Values")
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    save_plot(args=args, name="Parallel_Coordinate")
    optuna.visualization.matplotlib.plot_contour(study)
    save_plot(args=args, name="Contour")
    optuna.visualization.matplotlib.plot_slice(study)
    save_plot(args=args, name="Slice")
    optuna.visualization.matplotlib.plot_param_importances(study)
    save_plot(args=args, name="Hyperparameters_Importances")
    optuna.visualization.matplotlib.plot_edf(study)
    save_plot(args=args, name="Empirical Distribution Function")
    # Nota: Al parecer, "plot_rank" es experimental
    optuna.visualization.matplotlib.plot_rank(study)
    save_plot(args=args, name="Rank")
    optuna.visualization.matplotlib.plot_timeline(study)
    save_plot(args=args, name="Time_Plot")
# -- FIN save_plot -- #

# TODO: Por comprobar de que vaya bien en paralelo.
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
def objective(trial:optuna.Trial):
    
    args = ARGS

    # Ojo: Cambio importante respecto de la versión normal
    device = args[cons.DEVICE]        

    if args[cons.IS_DISTRIBUTED]:
        trial = optuna.integration.TorchDistributedTrial(trial)

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device: logger.info(f"Device: {device}")

    # Selección de los hiperparámetros de Optuna
    args[cons.MODEL] = trial.suggest_categorical("model", cons.POSSIBLE_MODELS)
    model = mr.load_model(args=args, device=device, is_main_device=is_main_device)

    if is_main_device: logger.info(f"Model: {model}")

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)            
    optimizer_name = trial.suggest_categorical("optimizer", cons.POSSIBLE_OPTIMIZERS)    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    if is_main_device: logger.info(f"Optimizer: {optimizer_name}, lr: {lr}")
    # --

    datasets = load_datasets(args, is_main_device)
    data_loaders = load_data_loaders(args, datasets)
    metricas = get_list_metrics(args[cons.NUMBER_CLASSES], device=device)

    if is_main_device: logger.info(f"{'-' * cons.NUM_GUIONES} Iniciando entrenamiento {'-' * cons.NUM_GUIONES}")

    loss_func = torch.nn.CrossEntropyLoss()    

    num_epochs = args[cons.EPOCS]
    # Entrenando
    for epoch in range(num_epochs):

        if is_main_device: logger.info(f"Epoch: {epoch + 1}/{num_epochs}")

        model.train(True)
        train_one_epoch(model=model, device=device, training_loader=data_loaders[cons.TRAIN_FOLDER_NAME],
                        loss_func=loss_func, optimizer=optimizer, estimacion_duracion=("", ""), 
                        is_main_device=is_main_device, debuging=args[cons.SHOW_DEBUG_OUTPUTS])

        results = mr.evaluate_model(model=model, dataloader=data_loaders[cons.VALIDATION_FOLDER_NAME], device=device,
                                 is_main_device=is_main_device, lista_metricas=metricas, args=args, 
                                 loss_fn=loss_func, save_confusion_matrix=False, nombre_prueba="Validation")
        
        trial.report(results[cons.MAIN_METRIC], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            

    if is_main_device: logger.info(f"{'-' * cons.NUM_GUIONES} Prueba finalizada {'-' * cons.NUM_GUIONES}")

    return results[cons.MAIN_METRIC]
# -- FIN objective -- #

def main_optuna(args:dict):    
    global ARGS

    ARGS = args

    is_main_device = ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)   

    # Ojo: cambio importante respecto de la versión normal
    device, _ = setup_gpu(args=args, logger=logger)            
    args[cons.DEVICE] = device
    
    # Por lo que veo aquí: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
    #   solo el hilo principal se encarga de poner en marcha la optimización y dentro es en donde se hace el paralelismo (entiendo)
    if is_main_device:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cons.OPTUNA_SEED),
            pruner=optuna.pruners.MedianPruner(),
            )
        
        # Para que la salida de optuna se guarde en el log.
        add_file_handler(loggers = [optuna.logging.get_logger("optuna")])
        set_level(loggers = [optuna.logging.get_logger("optuna")], level = cons.loggin_level)
        
        study.optimize(objective, n_trials = cons.OPTUNA_NUMBER_TRIALS, timeout = cons.OPTUNA_TIMEOUT)    

        show_and_save_results(study, args)
        info = '\n'.join([f"Number finished trials: {len(study.trials)}\n", 
                          f"Best trial:\n{study.best_trial}\n",
                          f"- {cons.MAIN_METRIC} value: {study.best_trial.value}\n",
                          f"- Parameters:\n{study.best_trial.params}"])                
        logger.info(info)
    else:
        # Mirar esto.
        for _ in range(cons.OPTUNA_NUMBER_TRIALS):
            try:        
                objective(None)
            except optuna.TrialPruned:
                pass

# -- FIN main_optuna -- #
