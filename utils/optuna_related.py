import optuna

import os
import matplotlib.pyplot as plt
import torch
import torch.distributed
from torchvision.transforms import v2

from torch.nn.parallel import DistributedDataParallel
from typing import Dict
from torchvision.datasets import ImageFolder

import utils.constants as cons
from utils.operations import GET_IMAGES_FOLDER_PATH
from utils.log_writer import getLogWritter, set_level, add_file_handler

from utils.training import train_one_epoch
import utils.model_related as mr
from utils.metrics import  get_list_metrics
from utils.data_loaders import load_datasets, load_data_loaders
from utils.operations import setup_gpu, cleanup

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

# TODO: Comprobar de que vaya
def modify_model_layers(model:torch.nn.Module, model_name:str, trial:optuna.Trial):

    mod_classifier_layer = list()

    classifier_layer = mr.get_classifier_layer(model, model_name)

    num_ftrs = classifier_layer.in_features
    if trial.suggest_categorical("add BatchNorm1D", [True, False]):
        mod_classifier_layer.append(torch.nn.BatchNorm1d(num_features=num_ftrs))
    
    if trial.suggest_categorical("add LayerNorm", [True, False]):
        mod_classifier_layer.append(torch.nn.LayerNorm(normalized_shape=num_ftrs))    

    if "vgg" in model_name:
        # Capas de Dropout (de base, p=0.5)
        model.classifier[2] = torch.nn.Dropout(p = trial.suggest_float(name="p droput", low=0, high=0.5, step=0.1))
        model.classifier[4] = torch.nn.Dropout(p = trial.suggest_float(name="p droput", low=0, high=0.5, step=0.1))

        mod_classifier_layer.append(model.classifier[6])

        model.classifier[6] = torch.nn.Sequential(*mod_classifier_layer)

    elif "resnet" in model_name:
        
        mod_classifier_layer.append(torch.nn.Dropout(p = trial.suggest_float(name="p droput", low=0, high=0.2, step=0.1)))
        mod_classifier_layer.append(model.fc)

        model.fc = torch.nn.Sequential(*mod_classifier_layer)
    elif "densenet" in model_name:

        mod_classifier_layer.append(torch.nn.Dropout(p = trial.suggest_float(name="p droput", low=0, high=0.2, step=0.1)))
        mod_classifier_layer.append(model.classifier)

        model.classifier = torch.nn.Sequential(*mod_classifier_layer)
# -- FIN modify_model_layers -- #

def modify_transformations(dict_datasets:Dict[str, ImageFolder], trial:optuna.Trial):

    dict_datasets[cons.TRAIN_FOLDER_NAME].transform = v2.Compose([
        v2.PILToTensor(),

        v2.Resize((trial.suggest_int(name="Tranform_Resize_altura", low=224, high=824, step=300), 
                   trial.suggest_int(name="Tranform_Resize_anchura", low=224, high=824, step=300))),
        
        v2.RandomHorizontalFlip(p=trial.suggest_float(name="Transform_RandomHorizontalFlip", low=0, high=1, step=0.2)),
        
        v2.RandomVerticalFlip(p=trial.suggest_float(name="Transform_RandomVerticalFlip", low=0, high=1, step=0.2)),
        
        v2.RandomPerspective(distortion_scale=trial.suggest_float(name="Transform_RandomPerspective", low=0, high=0.9, step=0.15)), 
        
        v2.RandomRotation(degrees=trial.suggest_int(name="Transform_RandomRotation", low=0, high=90, step=15)),

        v2.ColorJitter(brightness=trial.suggest_float(name="ColorJitter_brightness", low=0.0, high=1.0, step=0.1),
                       contrast=trial.suggest_float(name="ColorJitter_contrast", low=0.0, high=1.0, step=0.1),
                       saturation=trial.suggest_float(name="ColorJitter_saturation", low=0.0, high=1.0, step=0.1),
                       hue=trial.suggest_float(name="ColorJitter_hue", low=0.0, high=0.5, step=0.1)),

        v2.ToDtype(torch.float32, scale=True)
    ])
# -- FIN modify_transformations -- #


# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
def objective(trial:optuna.Trial):
        
    args = ARGS

    # Ojo: Cambio importante respecto de la versión normal
    device = args[cons.DEVICE]        

    if args[cons.IS_DISTRIBUTED]:
        # Nota: Al parecer, "TorchDistributedTrial" es experimental
        trial = optuna.integration.TorchDistributedTrial(trial)
        torch.distributed.barrier()

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device: logger.info(f"Device: {device}")

    datasets = load_datasets(args, is_main_device)

    # Selección de los hiperparámetros de Optuna
    args[cons.MODEL] = trial.suggest_categorical("model", cons.POSSIBLE_MODELS)
    model = mr.load_model(args=args, device=device, is_main_device=is_main_device)
    
    modify_model_layers(model=model, model_name=args[cons.MODEL], trial=trial)
    if is_main_device: logger.info(f"Model: {model}")    

    modify_transformations(datasets, trial)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", cons.POSSIBLE_OPTIMIZERS)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    args[cons.BATCH_SIZE] = trial.suggest_int("Batch size", 12, 48, step=6) # Esto se utiliza en "load_data_loaders"

    if is_main_device: logger.info(f"Optimizer: {optimizer_name}, lr: {lr}")
    # ---------

    model = model.to(device)
    if args[cons.IS_DISTRIBUTED]:
        model = DistributedDataParallel(model, device_ids=[device])        

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
    # ------

    if is_main_device: logger.info(f"{'-' * cons.NUM_GUIONES} Prueba finalizada {'-' * cons.NUM_GUIONES}")

    return results[cons.MAIN_METRIC]
# -- FIN objective -- #

def main_optuna(args:dict):    
    global ARGS

    ARGS = args

    is_main_device = ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)   

    # Cambio importante respecto de la versión normal
    device, _ = setup_gpu(args=args, logger=logger)            
    args[cons.DEVICE] = device
    
    # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
    if is_main_device:
        study = optuna.create_study(
            study_name="Optimización",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cons.OPTUNA_SEED),
            pruner=optuna.pruners.MedianPruner()
            #pruner=optuna.pruners.NopPruner()
            )

        # Para que la salida de optuna se guarde en el log.
        add_file_handler(loggers = [optuna.logging.get_logger("optuna")])
        set_level(loggers = [optuna.logging.get_logger("optuna")], level = cons.loggin_level)  

        study.optimize(objective, n_trials = cons.OPTUNA_NUMBER_TRIALS)

        show_and_save_results(study, args)
        info = '\n'.join([f"Number finished trials: {len(study.trials)}\n", 
                          f"Best trial:\n{study.best_trial}\n",
                          f"- {cons.MAIN_METRIC} value: {study.best_trial.value}\n",
                          f"- Parameters:\n{study.best_trial.params}"])                
        logger.info(info)
    else:
        # Nota: Si el hilo principal tiene un timeout y se finaliza por éste, los hilos secundarios esperan nuevas tareas del principal.
        for _ in range(cons.OPTUNA_NUMBER_TRIALS):
            try:        
                objective(None)
            except optuna.TrialPruned:
                pass
    
    if args[cons.IS_DISTRIBUTED]:
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.debug("- cleanup -")        
        cleanup()

# -- FIN main_optuna -- #
