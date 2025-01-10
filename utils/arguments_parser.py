import os

import argparse
import logging

import utils.constants as cons
from utils.log_writer import getLogWritter

from torch import Tensor

logger = getLogWritter(__name__)

# TODO: Poner la ayuda en un mismo idioma.
def get_args_parser():
    parser = argparse.ArgumentParser(prog="Proyecto", description="Modelo")
    
    parser.add_argument(f"--{cons.BATCH_SIZE}", default=64, type=int,
                    help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    
    parser.add_argument(f"--{cons.EPOCS}", default=50, type=int)

    parser.add_argument(f"--{cons.INPUT_SIZE}", default='524x224', type=str,
                    help=f"Usos:\n\t-Un solo número: imagen cuadrada.\n\tDos números separados por \'{cons.SEPARADOR_INPUT_IMAGENES}\': [altura de la imagen]{cons.SEPARADOR_INPUT_IMAGENES}[anchura de la imagen]"+
                         f"Ejemplos: 224 ==> Imagen cuadrada de 224x224. 512{cons.SEPARADOR_INPUT_IMAGENES}214: Imagen de 512 de altura y 214 de anchura"
                            )

    parser.add_argument(f"--{cons.LEARNING_RATE}", type=float, default=0.0000245, help="learning rate")

    parser.add_argument(f"--{cons.EARLY_STOPPING_PATIENCE}", type=int, default=-1, 
        help="Número de épocas sin suficientes cambios para que finalice el entrenamiento antes de tiempo (si es -1, está \"desactivado\").")
    parser.add_argument(f"--{cons.EARLY_STOPPING_MIN_DELTA}", type=float, default=0.5, 
        help="Diferencía mínima en las últimas etapas para que el contador del \"early stopper\" avance.") 
        
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
                help="Model's weights path")
    
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
                        help=f"""Los pesos de cada modelo para cada clase durante las votaciones. 
                        Los pesos de cada clase están separados por {cons.SEPARATOR_VOTATION_CLASS} y los modelos por {cons.SEPARATOR_VOTATION_MODEL}.
                        A cada clase se le asigna la lista de pesos de la votación en el mismo orden que se han pasado en el parámetro '--{cons.MODEL}'.
                        Ejemplo con 3 clases y dos modelos: --{cons.MODEL} \"vgg19{cons.SEPARADOR_ENSEMBLE}resnet50\". --{cons.ENSEMBLE_VOTATION_WEIGHTS} \"1{cons.SEPARATOR_VOTATION_CLASS}0.5{cons.SEPARATOR_VOTATION_CLASS}0.1{cons.SEPARATOR_VOTATION_MODEL}0.5{cons.SEPARATOR_VOTATION_CLASS}1.3{cons.SEPARATOR_VOTATION_CLASS}0.8\"
                        """)
    

    parser.add_argument(f"--{cons.ROTATION_DEGREES}", default=75, type=float, help=f"Grados de rotación máxima que pueden tener las imágenes.")
    parser.add_argument(f"--{cons.RANDOM_PERSPECTIVE_DISTORSION}", default=0.75, type=float, help=f"Escala de la distorsión de la transformación.")
    parser.add_argument(f"--{cons.P_HORIZONTAL_FLIP}", default=0.2, type=float, help=f"Probabilidad de que se realize un giro horizontal de la imagen.")
    parser.add_argument(f"--{cons.P_VERTICAL_FLIP}", default=0.6, type=float, help=f"Probabilidad de que se realize un giro vertical de la imagen.")

    # Variables relacionadas con "Color Jitter"
    parser.add_argument(f"--{cons.COLOR_JITTER_BRIGHTNESS}", default="1", type=str, 
                        help=f"No utilizar números negativos. Si es un solo número, el rango será: [max(0, 1 - N), 1 + N]. Si se dan dos número separados por \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', esos serán el mínimo y máximo: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_CONTRAST}", default="0.6", type=str, 
                        help=f"No utilizar números negativos. Si es un solo número, el rango será: [max(0, 1 - N), 1 + N]. Si se dan dos número separados por \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', esos serán el mínimo y máximo: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_SATURATION}", default="0.1", type=str, 
                        help=f"No utilizar números negativos. Si es un solo número, el rango será: [max(0, 1 - N), 1 + N]. Si se dan dos número separados por \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \', esos serán el mínimo y máximo: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"")
    parser.add_argument(f"--{cons.COLOR_JITTER_HUE}", default="0.4", type=str, 
                        help=f"Si es un solo número se debe escoger entre [0, 0.5] y el resultado será [-hue, hue]. Si son dos números separados por \' {cons.SEPARADOR_INPUTS_COLOR_JITTER} \' (por ejemplo: \"min{cons.SEPARADOR_INPUTS_COLOR_JITTER}max\"), escoger valores en el intervalo [-0.5, 0.5] y el resultado será un número en el intervalo [min, max]")

    parser.add_argument(f"--{cons.IS_HYPERTUNING}", action='store_true', default=False,
        help="True: Se va a realizar una búsqueda de hiperparámetros.")
    
    parser.add_argument(f"--{cons.OPTIMIZER}", default="Adam", type=str, 
                        help=f"Optimizer's name. Options: {str(cons.SWITCH_OPTIMIZERS.keys()).replace('[', '').replace(']', '')}")
    
    parser.add_argument(f"--{cons.LOSS_FUNCTION}", default="MSE", type=str, 
                    help=f"Loss function's name. Options: {str(cons.SWITCH_LOSS_FUNCTIONS.keys()).replace('[', '').replace(']', '')}")

    parser.add_argument(f"--{cons.P_DROPOUT}", default=0.0, type=float, help=f"Probabilidad de que la capa de dropout se active")

    parser.add_argument(f"--{cons.INFERENCE}", action='store_true', default=False, 
                        help=f"True si realizar una inferencia, False si no. Se utilizarán todas las imagenes disponibles en la ruta de los datos.")

    parser.add_argument(f"--{cons.IS_REGRESSION}", action='store_true', default=False,
                        help="Is a regression model instead of a classifier.")

    parser.add_argument(f"--{cons.REGRESSION_CLASS_BOUNDARIES}", type=str, default=f"0{cons.SEPARADOR_CLASS_BOUNDRIES}1{cons.SEPARADOR_CLASS_BOUNDRIES}2{cons.SEPARADOR_CLASS_BOUNDRIES}3{cons.SEPARADOR_CLASS_BOUNDRIES}4",
                        help="The boundries to convert the regression outputs into classification-like values. The numbers must be separated by '{cons.SEPARADOR_CLASS_BOUNDRIES}' and must be greater than the previous one. The number of boundries must coincide with the number of classes.")

    return parser
# -- Fin get_args_parser -- #

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
    print(f"models: {models}") # TODO: BORRAR
    args[cons.MODEL] = models
# -- Fin check_modelos -- #

def check_and_set_votation_weights(args:dict):
    votation_weights = args[cons.ENSEMBLE_VOTATION_WEIGHTS].split(cons.SEPARATOR_VOTATION_MODEL)

    num_modelos = len(args[cons.MODEL])
    if len(votation_weights) != num_modelos:
        texto = f"El número de pesos de las votación ({len(votation_weights)}) no coincide con el número de modelos({num_modelos}). "
        if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
            logger.error(texto)
        argparse.ArgumentError(None, texto)   

    args[cons.ENSEMBLE_VOTATION_WEIGHTS] = {args[cons.MODEL][i] : list(map(float, votation_weights[i].split(cons.SEPARATOR_VOTATION_CLASS))) 
                                            for i in range(num_modelos)}
    print(f"args[{cons.ENSEMBLE_VOTATION_WEIGHTS}]: {args[cons.ENSEMBLE_VOTATION_WEIGHTS]}") # TODO: BORRAR
# -- Fin check_votation_weights -- #

def check_and_set_debug_options(args:dict): 
    # Comprobamos si mostrar las opciones de debug
    if args[cons.SHOW_DEBUG_OUTPUTS]:
        cons.loggin_level = logging._levelToName[logging.DEBUG]
    #else: logging._levelToName(logging.INFO)
# -- Fin check_and_set_debug_options -- #

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

def check_and_set_regression_boundaries(args:dict):
        # Vamos a comprobar que los límites estén bien.
    if args[cons.IS_REGRESSION]:
        boundries = args[cons.REGRESSION_CLASS_BOUNDARIES].split(cons.SEPARADOR_CLASS_BOUNDRIES)

        num_boundries = len(boundries)

        if num_boundries < args[cons.NUMBER_CLASSES]:
            texto = f"The number of boundries is not equal to the number of classes (|{boundries}| != {args[cons.NUMBER_CLASSES]})."
            logger.error(texto)
            argparse.ArgumentError(None, texto)

        for i in range(num_boundries-1):
            if boundries[i] >= boundries[i+1]: 
                texto = f"Every number must be greater than the previous one {boundries}."
                logger.error(texto)
                argparse.ArgumentError(None, texto)
        
        args[cons.REGRESSION_CLASS_BOUNDARIES] = Tensor(list(map(int, boundries)))
# -- Fin check_and_set_regression_boundaries -- #

def check_args(args:dict):
    check_and_set_models(args)
    check_and_set_votation_weights(args)
    check_and_set_debug_options(args)
    check_and_set_transformations(args)
    check_and_set_split_percentages(args)
    set_list_no_augment_classes(args)
    check_and_set_regression_boundaries(args)
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
