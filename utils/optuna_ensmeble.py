import optuna

import os
import matplotlib.pyplot as plt
import torch
import torch.distributed

import utils.constants as cons
from utils.operations import GET_IMAGES_FOLDER_PATH
from utils.log_writer import getLogWritter
from utils.training import train_model
from utils.model_related import evaluate_model, load_model
from utils.operations import cleanup
from utils.log_writer import getLogWritter, set_level, add_file_handler

# --- Inicio constantes y variables globales --- #
logger = getLogWritter(__name__)

ARGS = None
LISTA_MODELOS_POSIBLES = None

RUTA_PESOS = "/home/pluijter/proyecto/Ensemble_test/pesos"

#MODELOS_CLASIFICACION = ["densenet169", "desnsenet121", "resnet50", "resnet152", "vgg19"]
MODELOS_CLASIFICACION = ["densenet169", "resnet50"]
MODELOS_REGRESION = ["densenet169", "resnet50"]
REGRESSION_LOSS_FUNCTIONS = ["MSE", "MAE"]
CLASIFICATION_LOSS_FUNCTIONS = ["CrossEntropyLoss"]
#MODELOS_REGRESION= ["0,1,2,3,4", "0.3,1.3,2.3,3.3,4","0.7,1.7,2.7,3.7,4", "0.7,1.7,2.5,3.5,4","0.8,1.8,2.3,3.3,4","0.8,1.8,2.8,3.3,4"]
LIMITES_REGRESION= ["0,1,2,3,4", "0.3,1.3,2.3,3.3,4", "0.7,1.7,2.7,3.7,4", "0.7,1.7,2.5,3.5,4", "0.8,1.8,2.8,3.3,4"]

OPTUNA_METRICAS = "optuna_metricas"
OPTUNA_METRICAS_REGRESION = "optuna_metricas_regresion"
OPTUNA_DATA_LOADERS = "optuna_data_loaders"
OPTUNA_PROPORCION_CLASES = "optuna_proporcion_clases"

MAX_MODELS = 3
MIN_WEIGHT_VOTATION = 0.5
MAX_WEIGHT_VOTATION = 1

# --- Fin Inicio constantes y variables globales --- #

def _generate_path_weights(model_name:str, loss_name:str, extension:str, reg_boundaries:str, ruta_pesos:str)->str:

    print(f"model_name: {model_name}")
    print(f"loss_name: {loss_name}")
    print(f"extension: {extension}")
    print(f"reg_boundaries: {reg_boundaries}")
    print(f"ruta_pesos: {ruta_pesos}")

    if reg_boundaries is None:
        output = os.path.join(ruta_pesos, "clasificacion")
        type_model = "model"        
    else:
        output = os.path.join(ruta_pesos, "regresion")
        output = os.path.join(output, reg_boundaries)
        type_model = "regression_model"
    loss_name = f"_{loss_name}" if loss_name in REGRESSION_LOSS_FUNCTIONS else ""
    
    name_file = f"{type_model}_{model_name}{loss_name}{extension}"

    return os.path.join(output, name_file)
# --- End _generate_path_weights --- #

# Para cada modelo, se genera la ruta a sus pesos y se guarda en una lista
def generate_path_weights(models:list, list_losses:list, boundaries:list, extension:str =".pth", ruta_pesos:str=RUTA_PESOS)->list:
    
    #rutas_pesos = list()
    #for i in range(len(models)):
    #    rutas_pesos.append() = _generate_path_weights(model_name=models[i], loss_name=list_losses, reg_boundaries=boundaries, extension=extension, ruta_pesos=ruta_pesos)

    return [_generate_path_weights(model_name=models[i], loss_name=list_losses[i], reg_boundaries=boundaries[i], extension=extension, ruta_pesos=ruta_pesos)
            for i in range(len(models))]
# -- generate_path_weights -- #

def _generate_votation_weights(num_clases:int, min_value:int, max_value:int, trial: optuna.Trial)->torch.Tensor:
    votation_weights = torch.zeros(num_clases)
    for i in range(num_clases):
        num_clases[i] = trial.suggest_float(f"votation_weight[{i}]", low= min_value, high= max_value)

    #return torch.Tensor([trial.suggest_float(f"votation_weight[{i}]", low= min_value, high= max_value) for i in range(num_clases)])

    print(f"votation_weights: {votation_weights}")

    return votation_weights
# -- Fin _generate_votation_weights -- #

def generate_votation_weights(num_clases:int, min_value:int, max_value:int, trial: optuna.Trial, types_model:list)->torch.Tensor:
    
    _types_model = list(filter(lambda x: x == cons.MODEL_TYPE_REGRESSION, types_model))
    print(f"_types_model: {_types_model}") # TODO: Borrar
    dict_types_model = {
        index : _generate_votation_weights(num_clases, min_value, max_value, trial)
        for index, _ in _types_model
    }
    print(f"dict_types_model: {dict_types_model}") # TODO: Borrar

    return dict_types_model
# -- Fin generate_votation_weights -- #

def generate_list_possible_models(list_class_models:list=MODELOS_CLASIFICACION, list_reg_models:list=MODELOS_REGRESION, 
                                  list_class_loss:list=CLASIFICATION_LOSS_FUNCTIONS, list_reg_loss:list=REGRESSION_LOSS_FUNCTIONS)-> list:
    
    print(f"list_class_models: {list_class_models}")
    print(f"list_class_loss: {list_class_loss}")
    print(f"list_reg_models: {list_reg_models}")
    print(f"list_reg_loss: {list_reg_loss}")

    class_models = [(model, loss, None) for model in list_class_models for loss in list_class_loss]
    #reg_models = [(model, loss, boundary) for model in list_reg_models for loss in list_reg_loss for boundary in LIMITES_REGRESION]    
    reg_models = list()
    for model in list_reg_models:
        for loss in list_reg_loss:
            if not (model == "densenet169" and loss == "MSE"):
                # La combinación "densenet169" y "MSE" no salió bien.
                for boundary in LIMITES_REGRESION:
                    reg_models.append((model, loss, boundary))   

    print(f"class_models: {class_models}")
    print(f"reg_models: {reg_models}")

    return class_models + reg_models
# -- Fin generate_list_possible_models -- #

def _generate_list_models_ensemble(trial: optuna.Trial, list_posible_models:list, max_modelos:int)->list:
    
    modelos_posibles = list_posible_models.copy()
    modelos_escogidos = list()
    #for elem in modelos_posibles:
    #    print(f"type: {type(elem)}|| elem: {elem}")

    for i in range(max_modelos):
        modelo = trial.suggest_categorical(f"model, loss and boundaries - number: {i}", modelos_posibles)
        modelos_escogidos.append(modelo)
        modelos_posibles.remove(modelo)
    
    return modelos_escogidos
# -- Fin _generate_list_models_ensemble -- #

def generate_list_models_ensemble(trial: optuna.Trial, list_posible_models:list=LISTA_MODELOS_POSIBLES, max_modelos:int=MAX_MODELS)->tuple:
    
    modelos_escogidos = _generate_list_models_ensemble(trial=trial, list_posible_models=list_posible_models, max_modelos=max_modelos)

    #modelos = [model for model,_,__ in modelos_escogidos]
    #tipos_modelos = [ (cons.MODEL_TYPE_REGRESSION if loss in REGRESSION_LOSS_FUNCTIONS else cons.MODEL_TYPE_CLASSIFIER) for _, loss, __ in modelos_escogidos]
    #boundaries = [ (limite if loss in REGRESSION_LOSS_FUNCTIONS else None) for _, loss, limite in modelos_escogidos]
    
    modelos, tipos_modelos, boundaries, loss_functions = list(), list(), dict(), list()
    
    # TODO: los boundaries no eran un diccionario ?? --> Revisarlo y corregirlo

    i = 0
    for model, loss, limite in modelos_escogidos:
        modelos.append(model)
        loss_functions.append(loss)

        if loss in REGRESSION_LOSS_FUNCTIONS:
            tipos_modelos.append(cons.MODEL_TYPE_REGRESSION)
            boundaries[i] = (limite)
        else:
            tipos_modelos.append(cons.MODEL_TYPE_CLASSIFIER)

        i += 1

    return (modelos, tipos_modelos, boundaries, loss_functions)
# -- Fin _generate_list_models_ensemble -- #

def save_plot(name:str):
    
    path_images_folder = GET_IMAGES_FOLDER_PATH()

    if not os.path.isdir(path_images_folder):
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

# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
def objective(trial:optuna.Trial):
    global ARGS, LISTA_MODELOS_POSIBLES
    args = ARGS
    print(f"LISTA_MODELOS_POSIBLES [objective]: {LISTA_MODELOS_POSIBLES}")

    device = args[cons.OPTUNA_DEVICES]

    if args[cons.IS_DISTRIBUTED]:
        # Nota: Al parecer, "TorchDistributedTrial" es experimental
        trial = optuna.integration.TorchDistributedTrial(trial)
        torch.distributed.barrier()

    is_main_device = (args[cons.IS_DISTRIBUTED] and device == 0) or not args[cons.IS_DISTRIBUTED]

    if is_main_device: logger.info(f"Device: {device}")

    metricas = args[OPTUNA_METRICAS]
    metricas_regresion = args[OPTUNA_METRICAS_REGRESION]
    data_loaders = args[OPTUNA_DATA_LOADERS]
    proporcion_clases = args[OPTUNA_PROPORCION_CLASES]

    # Listas de los modelos, tipos de modelos y los limites para la regresión seleccionados
    args[cons.MODEL], args[cons.ENSEMBLE_TYPE_MODEL], args[cons.REGRESSION_CLASS_BOUNDARIES], loss_functions = generate_list_models_ensemble(trial=trial, list_posible_models=LISTA_MODELOS_POSIBLES)    
    

    if args[cons.INFERENCE]:
        # Si se va a hacer una inferencia, se buscan los pesos de las votaciones para cada combinación:
        args[cons.ENSEMBLE_VOTATION_WEIGHTS] = generate_votation_weights(num_clases=args[cons.NUMBER_CLASSES], min_value=MIN_WEIGHT_VOTATION, 
                                                                         max_value=MAX_WEIGHT_VOTATION, trial=trial, types_model=args[cons.ENSEMBLE_TYPE_MODEL])
    else:
        # Si se va a entrenar la capa de clasificación:
        args[cons.IS_REGRESSION] = trial.suggest_categorical("Regression_classifier", [True, False])        
        args[cons.OPTIMIZER] = trial.suggest_categorical("Optimizer", cons.SWITCH_OPTIMIZERS)
        args[cons.LOSS_FUNCTION] = trial.suggest_categorical("Loss Function", REGRESSION_LOSS_FUNCTIONS if args[cons.IS_REGRESSION] 
                                                             else CLASIFICATION_LOSS_FUNCTIONS)

        if is_main_device: logger.info(f"Regression_classifier: {args[cons.IS_REGRESSION]}\nOptimizer: {args[cons.OPTIMIZER]}")    
    
    # Generamos el listado de pesos
    args[cons.MODEL_WEIGHTS] = generate_path_weights(models=args[cons.MODEL],list_losses=loss_functions,boundaries=args[cons.REGRESSION_CLASS_BOUNDARIES])

    # Generamos el ensemble
    model = load_model(args=args, device=device, is_main_device=is_main_device)

    print(f"SWITCH_LOSS_FUNCTIONS: {cons.SWITCH_LOSS_FUNCTIONS.keys()}") # TODO: Borrar

    if args[cons.INFERENCE]:
        evaluate_model(model=model, dataloader=data_loaders, device=device, 
                       is_main_device=is_main_device, lista_metricas=metricas, 
                       args=args, metricas_regression=metricas_regresion)
    else:
        train_model(args=args, model=model, dataloaders=data_loaders,
                    is_main_device=is_main_device, device=device, lista_metricas=metricas, 
                    proporcion_clases=proporcion_clases, metricas_regresion=metricas_regresion, 
                    optuna_trial=trial)
# -- Fin objective -- #


# Esta función se tiene que ejecutar antes de cargar los pesos (y después de lo de los datasets) en el main de verdad
def main_optuna(args:dict, metricas:list, metricas_regresion:list, data_loaders:dict, proporcion_clases:torch.Tensor):
                    
    global ARGS, LISTA_MODELOS_POSIBLES
    ARGS = args

    args[OPTUNA_METRICAS] = metricas
    args[OPTUNA_METRICAS_REGRESION] = metricas_regresion
    args[OPTUNA_DATA_LOADERS] = data_loaders
    args[OPTUNA_PROPORCION_CLASES] = proporcion_clases

    LISTA_MODELOS_POSIBLES = generate_list_possible_models()
    print(f"LISTA_MODELOS_POSIBLES: {LISTA_MODELOS_POSIBLES}")

    is_main_device = ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)   

    #device, _ = setup_gpu(args=args, logger=logger)            
    #args[cons.DEVICE] = device
    
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

        logger.info(f"n_trials: {cons.OPTUNA_NUMBER_TRIALS}")
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
# -- Fin main_optuna -- #

# TODO: 
#   - AMBOS:
#       - Listado de modelos
#           -> Hay 5 modelos de clasificación.
#           -> Hay "3" modelos de regresión (densenetMAE y Resnet{MAE, MSE}), pero con 6 combinaciones de límites ==> 18
#           -> Una época tarda 0,5 horas en entrenarse (aprox) ==> Si entrenamos 5 épocas: 2,5 horas por combinación.
#       
#   - POR PESOS:  
#       - Cambio de pesos de la votación <= Esto no es un problema, hay muchas combinaciones pero las inferencias duran un minuto
#   
#   - POR "CLASIFICADOR":
#       - Probar por clasificación y regresión (y ambas funciones de pérdida) ==> *3 el número de pruebas.

