import os

import argparse
import logging

import utils.constants as cons
from utils.log_writer import getLogWritter

from torch import Tensor

logger = getLogWritter(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser(prog="Proyecto", description="Modelo")
    
    parser.add_argument(f"--{cons.BATCH_SIZE}", default=64, type=int,
                        help="Batch size per GPU")
                    #help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)")
    
    parser.add_argument(f"--{cons.EPOCS}", default=50, type=int,
                        help="Number of epochs")

    parser.add_argument(f"--{cons.INPUT_SIZE}", default='524x224', type=str,
                    help=f"Uses:\n\t-One number: square image.\n\tTwo numbers separted by \'{cons.SEPARADOR_INPUT_IMAGENES}\': [image's height]{cons.SEPARADOR_INPUT_IMAGENES}[image's width]"+
                         f"Examples: 224 ==> 224x224 square image. 512{cons.SEPARADOR_INPUT_IMAGENES}214: image of 512 pixels of height and 214 pixels of width"
                            )

    parser.add_argument(f"--{cons.LEARNING_RATE}", type=float, default=0.01, 
                        help="learning rate")
    
    parser.add_argument(f"--{cons.REDUCE_ON_PLATEAU_PATIENCE}", type=int, default=-1, 
                        help="Number of epochs without improvement before the learning rate is reduced")
    
    parser.add_argument(f"--{cons.REDUCE_ON_PLATEAU_RATE_FACTOR}", type=float, default=0.1, 
                        help="Number of epochs without improvement before the learning rate is reduced")

    parser.add_argument(f"--{cons.REDUCE_ON_PLATEAU_RATE_THRESHOLD}", type=float, default=1e-4, 
                        help="Number of epochs without improvement before the learning rate is reduced")

    parser.add_argument(f"--{cons.EARLY_STOPPING_PATIENCE}", type=int, default=-1, 
                        help="Number of epochs without enought changes to early finish the traning (if it is -1, it is deactivated)")
    parser.add_argument(f"--{cons.EARLY_STOPPING_MIN_DELTA}", type=float, default=0.5, 
                        help="Minimum difference between epoch to make the \"early stopper\"\'s counter to advance.") 
        
    parser.add_argument(f"--{cons.NUMBER_CLASSES}", default=5, type=int,
                        help="number of the classification types")

    parser.add_argument(f"--{cons.DATA_PATH}", default=None, type=str,
                        help=f"dataset path if you have all the data inside this folder, both separated in 3 folders ({cons.TEST_FOLDER_NAME}, {cons.VALIDATION_FOLDER_NAME}, {cons.TRAIN_FOLDER_NAME}) or all together, or if you have the train and validation data in that folder.")

    parser.add_argument(f"--{cons.TEST_DATA_PATH}", default=None, type=str,
                        help="Test dataset's path")
    parser.add_argument(f"--{cons.TRAIN_DATA_PATH}", default=None, type=str,
                        help="Test dataset's path")
    parser.add_argument(f"--{cons.VALIDATION_DATA_PATH}", default=None, type=str,
                        help="Validation dataset's path")
    parser.add_argument(f"--{cons.SPLIT_PERCENTAGES}", default=None, type=str,
                        help=f"If you have all the data in 1 folder: Train, validation and test percentajes. \
                        \"Train{cons.SEPARADOR_SPLIT_PERCENTAGES}Validation{cons.SEPARADOR_SPLIT_PERCENTAGES}Test\". \
                        If you have the train and validation data in one folder: \"Train{cons.SEPARADOR_SPLIT_PERCENTAGES}Validation\".")                        

    parser.add_argument(f"--{cons.OUTPUT_DIR}", default="./output_dir", type=str,
                        help="path where to save, empty for no saving")
    
    parser.add_argument(f"--{cons.DEVICE}", default="cuda", type=str,  
                        help="device to use for training / testing")
    
    parser.add_argument(f"--{cons.MODEL}", default="vgg19", type=str,
                        help=f"The model's name or, if it's a ensemble, a list of models separated by {cons.SEPARADOR_ENSEMBLE}. Example: vgg19{cons.SEPARADOR_ENSEMBLE}resnet50")
    
    parser.add_argument(f"--{cons.PARTIAL_MODELS_PATH}", default="./partial_models", type=str,
                        help="Path to the folder where the best models will be stored")    

    parser.add_argument(f"--{cons.MODEL_WEIGHTS}", default=None, type=str,
                        help=f"Model's weights path. In case an ensemble is going to be used, the different paths must be separated by: {cons.SEPARADOR_WEIGTHS_PATHS}. It is assumed that the weights' path is in the same order as the models passed as parameters.")
    
    parser.add_argument(f"--{cons.NO_DATA_AUGMENT_CLASSES}", default='0', type=str,
                        help=f"List of classes that will not be augmented separated by a \"{cons.SEPARADOR_NO_DATA_AUGMENT_CLASSES}\".")

    parser.add_argument(f"--{cons.BATCH_AUGMENTATION}", default=0, type=int,
                        help=f"Número de veces que se reptien las muestras en un mismo batch. Se omiten las clases que aparecen en: \"{cons.SEPARADOR_NO_DATA_AUGMENT_CLASSES}\".")

    parser.add_argument(f"--{cons.SHOW_DEBUG_OUTPUTS}", action='store_true', default=False,
                        help="Show the debug outputs")
    
    parser.add_argument(f"--{cons.NOT_FREEZE_LAYERS}", default=-1, type=int,
                        help="Number of the last layers to not freeze their learning. If -1: the model will work as normal.")
    
    parser.add_argument(f"--{cons.IS_DISTRIBUTED}", action='store_true', default=False,
                        help="True: the training is distributed. False: the training is only local.")

    parser.add_argument(f"--{cons.ENSEMBLE_VOTATION_WEIGHTS}", default="1,1,1,1,1", 
                        help= f"The weights of every model during the ensemble's votations.\n \
                        The weights will be seperated by {cons.SEPARATOR_VOTATION_CLASS} and the models by {cons.SEPARATOR_VOTATION_MODEL}.\n \
                        A votation weight will be assigned to every class in the same order as they have been passed in the parameter \'--{cons.MODEL}\'.\n\
                        Example of 3 classes and two models: --{cons.MODEL} \"vgg19{cons.SEPARADOR_ENSEMBLE}resnet50\". --{cons.ENSEMBLE_VOTATION_WEIGHTS} \"1{cons.SEPARATOR_VOTATION_CLASS}0.5{cons.SEPARATOR_VOTATION_CLASS}0.1{cons.SEPARATOR_VOTATION_MODEL}0.5{cons.SEPARATOR_VOTATION_CLASS}1.3{cons.SEPARATOR_VOTATION_CLASS}0.8\"")
        
    parser.add_argument(f"--{cons.ROTATION_DEGREES}", default=75, type=float, 
                        help=f"Maximum rotation degrees.")                        
    parser.add_argument(f"--{cons.RANDOM_PERSPECTIVE_DISTORSION}", default=0.75, type=float, 
                        help=f"Transformation's distorsion scale")                        
    parser.add_argument(f"--{cons.P_HORIZONTAL_FLIP}", default=0.2, type=float, 
                        help=f"Probability to apply a horizontal flip")    
    parser.add_argument(f"--{cons.P_VERTICAL_FLIP}", default=0.6, type=float, 
                        help=f"Probability to apply a vertical flip")
    

    # Variables relacionadas con "Color Jitter"
    parser.add_argument(f"--{cons.COLOR_JITTER_BRIGHTNESS}", default="1", type=str, 
                        help=f"Only non-negative numbers. If it's only a number, the range will be: [max(0, 1 - N), 1 + N]. If the input are two numbers separated by \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', those will be the minimum and the maximum: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_CONTRAST}", default="0.6", type=str, 
                        help=f"Only non-negative numbers. If it's only a number, the range will be: [max(0, 1 - N), 1 + N]. If the input are two numbers separated by \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', those will be the minimum and the maximum: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_SATURATION}", default="0.1", type=str, 
                        help=f"Only non-negative numbers. If it's only a number, the range will be: [max(0, 1 - N), 1 + N]. If the input are two numbers separated by \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', those will be the minimum and the maximum: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_HUE}", default="0.4", type=str, 
                        help=f"If the input it is only a number, it must be bewteen [0, 0.5] and the result will be [-hue, hue]. If the input are two numbers separated by \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \' (e.g.: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"), a value between [-0.5, 0.5] must be choosen and the result will be between [min, max]")

    parser.add_argument(f"--{cons.IS_HYPERTUNING}", action='store_true', default=False,
                        help="True: to make a hiperparameter search.")
    
    parser.add_argument(f"--{cons.OPTIMIZER}", default="Adam", type=str, 
                        help=f"Optimizer that will be used during training. Options: {str(cons.SWITCH_OPTIMIZERS.keys()).replace('[', '').replace(']', '')}")
    
    parser.add_argument(f"--{cons.LOSS_FUNCTION}", default="MSE", type=str, 
                        help=f"Loss function's name. Options: {str(cons.SWITCH_LOSS_FUNCTIONS.keys()).replace('[', '').replace(']', '')}")

    parser.add_argument(f"--{cons.P_DROPOUT}", default=0.0, type=float, 
                        help=f"Dropout layer's activation probability")    

    parser.add_argument(f"--{cons.INFERENCE}", action='store_true', default=False, 
                        help=f"False: Training. True: inference; for this, all images stored on dataset's path will be used.")    

    parser.add_argument(f"--{cons.IS_REGRESSION}", action='store_true', default=False,
                        help="Is a regression model instead of a classifier.")

    parser.add_argument(f"--{cons.REGRESSION_CLASS_BOUNDARIES}", type=str, default=f"0{cons.SEPARADOR_CLASS_BOUNDRIES}1{cons.SEPARADOR_CLASS_BOUNDRIES}2{cons.SEPARADOR_CLASS_BOUNDRIES}3{cons.SEPARADOR_CLASS_BOUNDRIES}4",
                        help=f"The boundries to convert the regression outputs into classification-like values. The numbers must be separated by '{cons.SEPARADOR_CLASS_BOUNDRIES}' and must be greater than the previous one. The number of boundries must coincide with the number of classes. In the case of creating an Ensemble, the number of set of boundaries must be equal to the number of regression models or, if a new regresion layer is going to be created, must be one more boundaries than regression models (this last one will be for the new last layer).")
    
    parser.add_argument(f"--{cons.ENSEMBLE_TYPE_MODEL}", type=str, default="classifier",
                        help=f"The ensemble models' types. It is a list of models separated by '{cons.SEPARATOR_ENSEMBLE_TYPE_MODEL}'. The number of model's type must coincide with the number of models. The model's types are: {cons.ENSEMBLE_MODEL_TYPES}")
    
    parser.add_argument(f"--{cons.ENSEMBLE_CREATE_CLASSIFIER}", action='store_true', default=False,
                        help="Create a classifier layer instead of using votation weights for ensemble models")

    parser.add_argument(f"--{cons.ENSEMBLE_CLASSIFIER_WEIGHTS}", default=None, type=str,
                        help=f"Ensemble's classifier layer's weights path.")

    return parser
# -- Fin get_args_parser -- #

def check_and_set_type_models(args:dict):
    
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info(f"args[{cons.ENSEMBLE_TYPE_MODEL}]: {args[cons.ENSEMBLE_TYPE_MODEL]}")
    
    model_types = args[cons.ENSEMBLE_TYPE_MODEL].split(cons.SEPARATOR_ENSEMBLE_TYPE_MODEL)

    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info(f"model_types: {model_types}")
    
    num_modelos = len(args[cons.MODEL])

    if len(model_types) != num_modelos:
        texto = f"El número de tipo de modelos ({len(args[cons.ENSEMBLE_TYPE_MODEL])}) no coincide con el número de modelos ({num_modelos})."
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.error(texto)

    # Si tenemos dos modelos iguales, uno machaca al otro en el diccionario.
    #model_types = {model_types[i] for i in range(num_modelos)}
    #if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
    #    logger.info(f"model_types (diccionario): {model_types}")

    args[cons.ENSEMBLE_TYPE_MODEL] = model_types
# -- Fin check_and_set_type_models -- #

def set_weights_paths(args:dict):
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(f"args[{cons.MODEL_WEIGHTS}]: {args[cons.MODEL_WEIGHTS]}")

    if args[cons.MODEL_WEIGHTS] is not None:
        weights = args[cons.MODEL_WEIGHTS].split(cons.SEPARADOR_WEIGTHS_PATHS)
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.debug(f"weights: {weights}")
    
        num_modelos = len(args[cons.MODEL])
        if len(weights) != num_modelos:
            texto = f"El número de rutas a los pesos ({len(weights)}) no coincide con el número de modelos ({num_modelos})."
            if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
                logger.error(texto)
            argparse.ArgumentError(None, texto)
    
        args[cons.MODEL_WEIGHTS] = weights
# -- Fin set_weights_paths -- #


def check_and_set_models(args:dict):
    # Comprobamos el modelo (o modelos si son unensemble)
    models = args[cons.MODEL].split(cons.SEPARADOR_ENSEMBLE)
    for model in models:
        if model not in cons.POSSIBLE_MODELS:
            texto = f"{model}. Received: {args[cons.MODEL]}. Expected one of {str(cons.POSSIBLE_MODELS)}"
            # ("LOCAL_RANK" not in os.environ) es true si se trabaja sin concurrencia
            if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):        
                logger.error(texto)
            argparse.ArgumentError(None, texto)    
    args[cons.MODEL] = models
# -- Fin check_and_set_models -- #

def check_and_set_votation_weights(args:dict):
    votation_weights = args[cons.ENSEMBLE_VOTATION_WEIGHTS].split(cons.SEPARATOR_VOTATION_MODEL)

    num_modelos = len(args[cons.MODEL])
    if len(votation_weights) != num_modelos:
        texto = f"El número de pesos de las votación ({len(votation_weights)}) no coincide con el número de modelos ({num_modelos})."
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.error(texto)
        argparse.ArgumentError(None, texto)   

    args[cons.ENSEMBLE_VOTATION_WEIGHTS] = [Tensor(list(map(float, votation_weights[i].split(cons.SEPARATOR_VOTATION_CLASS)))) 
                                            for i in range(num_modelos)]
# -- Fin check_and_set_votation_weights -- #

def set_debug_options(args:dict): 
    # Comprobamos si mostrar las opciones de debug
    if args[cons.SHOW_DEBUG_OUTPUTS]:
        cons.loggin_level = logging._levelToName[logging.DEBUG]
    #else: logging._levelToName(logging.INFO)
# -- Fin set_debug_options -- #

def check_and_set_transformations(args:dict):
    # Ponemos como toca la altura y la anchura
    sizes = args[cons.INPUT_SIZE].split(cons.SEPARADOR_INPUT_IMAGENES)    
    args[cons.ALTURA_IMG] = int(sizes[0])
    args[cons.ANCHURA_IMG] = int(sizes[1] if len(sizes) > 1 else sizes[0])

    # Ponemos como toca las variables de color jitter
    for color_jitter_op in [cons.COLOR_JITTER_BRIGHTNESS, cons.COLOR_JITTER_CONTRAST, cons.COLOR_JITTER_SATURATION, cons.COLOR_JITTER_HUE]:
        splitted = args[color_jitter_op].split(cons.SEPARADOR_INPUTS_COLOR_JITTER)
        args[color_jitter_op] = (float(splitted[0]), float(splitted[1])) if len(splitted) > 1 else float(splitted[0])
# -- Fin check_and_set_transformations -- #

def check_and_set_split_percentages(args:dict):
    # Poniendo como tocan los porcentajes
    if args[cons.SPLIT_PERCENTAGES] is None:
        args[cons.SPLIT_PERCENTAGES] = cons.DEFAULT_TRAIN_VAL_TEST_PERCENTAGES
    else:        
        args[cons.SPLIT_PERCENTAGES] = args[cons.SPLIT_PERCENTAGES].split(cons.SEPARADOR_SPLIT_PERCENTAGES)
    
    # Ahora poniéndolos en "Tanto por 1"
    for i in range(len(args[cons.SPLIT_PERCENTAGES])):
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.debug(f"Antes: {args[cons.SPLIT_PERCENTAGES][i]}")

        args[cons.SPLIT_PERCENTAGES][i] = float(args[cons.SPLIT_PERCENTAGES][i])/100

        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.debug(f"Después: {args[cons.SPLIT_PERCENTAGES][i]}")
# -- Fin check_and_set_split_percentages -- #

def set_list_no_augment_classes(args:dict):
    # Generamos una lista de las clases que no hay que aumentar
    args[cons.NO_DATA_AUGMENT_CLASSES] = args[cons.NO_DATA_AUGMENT_CLASSES].split(cons.SEPARADOR_NO_DATA_AUGMENT_CLASSES)
# -- Fin set_list_no_augment_classes -- #

def check_optimizer(args:dict):
    if args[cons.OPTIMIZER] not in cons.SWITCH_OPTIMIZERS.keys():
        texto = f"{args[cons.OPTIMIZER]}. Received: {args[cons.OPTIMIZER]}. Expected one of {str(cons.SWITCH_OPTIMIZERS.keys())}"
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.error(texto)
        argparse.ArgumentError(None, texto)
# -- Fin check_optimizer -- #    

#def _check_and_set_regression_boundaries(set_boundries:str, number_regression_model:int, number_classes:int=5)->Tensor:
def _check_and_set_regression_boundaries(set_boundries:str, number_classes:int=5) -> Tensor:
    rangos = set_boundries.split(cons.SEPARADOR_CLASS_BOUNDRIES)
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(f"rangos: {rangos}")
    num_boundries = len(rangos)

    if num_boundries != number_classes:
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            texto = f"The number of boundries is not equal to the number of classes (|{rangos}| != {number_classes})."
            logger.error(texto)
        argparse.ArgumentError(None, texto)

    for j in range(num_boundries-1):
        if rangos[j] >= rangos[j+1]: 
            if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
                texto = f"Every number must be greater than the previous one {rangos}."
                logger.error(texto)
            argparse.ArgumentError(None, texto)

    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(f"rangos: {rangos}")
        #logger.debug(f"number_regression_model: {number_regression_model}") 

    return Tensor(list(map(float, rangos))) 
# -- Fin _check_and_set_regression_boundaries -- #

def check_and_set_regression_boundaries(args:dict):
    
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(f"args[cons.ENSEMBLE_TYPE_MODEL]: {args[cons.ENSEMBLE_TYPE_MODEL]}")

    list_regression_models = list(filter(lambda tuple: tuple[1] == cons.MODEL_TYPE_REGRESSION,
                                enumerate(args[cons.ENSEMBLE_TYPE_MODEL])))

    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info(f"list_regression_models: {list_regression_models}")

    number_models = len(list_regression_models)
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(f"number_models: {number_models}| range(number_models): {range(number_models)}")

    set_boundries = args[cons.REGRESSION_CLASS_BOUNDARIES].split(cons.SEPARADOR_SPLIT_CLASS_BOUNDRIES)
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info(f"set_boundries: {set_boundries}")

    num_set_boundries = len(set_boundries)
    is_a_regression_ensemble_with_classifier = args[cons.IS_REGRESSION] and args[cons.ENSEMBLE_CREATE_CLASSIFIER] and (num_set_boundries == (number_models + 1))

    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info(f"num_set_boundries: {num_set_boundries} | number_models + 1: {number_models + 1} | args[cons.ENSEMBLE_CREATE_CLASSIFIER]: {args[cons.ENSEMBLE_CREATE_CLASSIFIER]}")
        logger.info(f"is_a_regression_ensemble_with_classifier: {is_a_regression_ensemble_with_classifier}")

    # Es incorecto SI:
    #   A) el número de umbrales no es igual al número de modelos de regresión 
    #   Y
    #   B) el número de umbrales no supera solo en 1 y al número de modelos de regresión y es un ensemble
    if num_set_boundries != number_models or not is_a_regression_ensemble_with_classifier:
        texto = f"The number of set of boundries is not equal to the number of classes (|{num_set_boundries}| != {number_models})."
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.error(texto)
        argparse.ArgumentError(None, texto)
    
    # Vamos a comprobar que los límites estén bien.
    # list_regression_models[i][0]: Posición en la lista de modelos del <<Modelo número "i">>
    boundries = {list_regression_models[i][0]: 
                 _check_and_set_regression_boundaries(set_boundries=set_boundries[i], number_classes=args[cons.NUMBER_CLASSES]) 
                 for i in range(number_models)}
    
    if is_a_regression_ensemble_with_classifier:
        boundries[cons.BOUNDARIES_ENSEMBLE_REGRESSION_CLASSIFIER] = _check_and_set_regression_boundaries(set_boundries=set_boundries[number_models], 
                                                                                                         number_classes=args[cons.NUMBER_CLASSES])
        
    args[cons.REGRESSION_CLASS_BOUNDARIES] = boundries
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.info(f"args[cons.REGRESSION_CLASS_BOUNDARIES]: {args[cons.REGRESSION_CLASS_BOUNDARIES]}")
    
# -- Fin check_and_set_regression_boundaries -- #

def check_args(args:dict):    
    check_and_set_models(args)
    check_and_set_votation_weights(args)
    set_weights_paths(args)
    set_debug_options(args)
    check_and_set_transformations(args)
    check_and_set_split_percentages(args)
    set_list_no_augment_classes(args)
    check_and_set_type_models(args)
    check_and_set_regression_boundaries(args)
    check_optimizer(args)
# -- Fin check_args -- #


def get_dict_args():    
     # ("LOCAL_RANK" not in os.environ) es true si se trabaja sin concurrencia
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug("get_dict_args")
    args = vars(get_args_parser().parse_args())    
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug("check_args")
    check_args(args)
    
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.info("Arguments:" + 
                    '\n'.join([(f"\t{k}: {args[k]}") for k in args]) +
                    '\n' + ('-' * cons.NUM_GUIONES)
                    )
    return args
# -- Fin get_dict_args -- #

#test
if __name__ == "__main__":
    dict_params = get_dict_args()

    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.debug(dict_params)
        for k in dict_params:
            logger.debug(f"dict_params[{k}]: {dict_params[k]}")
