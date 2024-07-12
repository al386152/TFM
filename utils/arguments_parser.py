import argparse
import logging

import utils.constants as cons
from utils.log_writer import getLogWritter

logger = getLogWritter(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser(prog="Proyecto", 
                                     description="Modelo")
    
    parser.add_argument(f"--{cons.BATCH_SIZE}", default=64, type=int,
                    help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    
    parser.add_argument(f"--{cons.EPOCS}", default=50, type=int)

    parser.add_argument(f"--{cons.INPUT_SIZE}", default='224', type=str,
                    help=f"Usos:\n\t-Un solo número: imagen cuadrada.\n\tDos números separados por \'{cons.SEPARADOR_INPUT_IMAGENES}\': [altura de la imagen]{cons.SEPARADOR_INPUT_IMAGENES}[anchura de la imagen]"+
                         f"Ejemplos: 224 ==> Imagen cuadrada de 224x224. 512{cons.SEPARADOR_INPUT_IMAGENES}214: Imagen de 512 de altura y 214 de anchura"
                            )
                           
    
    parser.add_argument(f"--{cons.DROP_PATH}", type=float, default=0.1,
                        help="Drop path rate (default: 0.1)")
    
    parser.add_argument(f"--{cons.WEIGTH_DECAY}", type=float, default=0.05,
                        help="weight decay (default: 0.05)")
    
    parser.add_argument(f"--{cons.LEARNING_RATE}", type=float, default=None,
                        help="learning rate (absolute lr)")
    
    parser.add_argument(f"--{cons.BASE_LEARNING_RATE}", type=float, default=1e-3,
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    
    parser.add_argument(f"--{cons.WARMUP_EPOCHS}", type=int, default=10, help="epochs to warmup LR")
        
    parser.add_argument(f"--{cons.NUMBER_CLASSES}", default=5, type=int,
                        help="number of the classification types")

    parser.add_argument(f"--{cons.DATA_PATH}", default="../Datasets/MESSIDOR2", type=str,
                        help="dataset path")
    parser.add_argument(f"--{cons.OUTPUT_DIR}", default="./output_dir", type=str,
                        help="path where to save, empty for no saving")
    
    parser.add_argument(f"--{cons.DEVICE}", default="cuda", type=str,  
                        help="device to use for training / testing")
    
    parser.add_argument(f"--{cons.MODEL}", default="vgg19", type=str,
                        help="The model's name")
                        #help="Path to the local model or the model's name")
    
    parser.add_argument(f"--{cons.PARTIAL_MODELS_PATH}", default="./partial_models", type=str,
                    help="Path to the folder where the best models will be stored")    

    parser.add_argument(f"--{cons.MODEL_WEIGHTS}", default=None, type=str,
                help="Model's weights path")
    
    parser.add_argument(f"--{cons.SHOW_DEBUG_OUTPUTS}", action='store_true', default=False,
            help="Show the debug outputs")
    
    parser.add_argument(f"--{cons.NOT_FREEZE_LAYERS}", default=-1, type=int,
            help="Number of the last layers to not freeze their learning. If -1: the model will work as normal.")
    
    parser.add_argument(f"--{cons.IS_DISTRIBUTED}", action='store_true', default=False,
            help="True: the training is distributed. False: the training is only local.")

    return parser

def check_args(args:dict):

    # Comprobamos el modelo
    if args[cons.MODEL] not in cons.POSSIBLE_MODELS:
        texto = f"{cons.MODEL}. Received: {args[cons.MODEL]}. Expected one of {str(cons.POSSIBLE_MODELS)}"
        logger.error(texto)
        argparse.ArgumentError(None, texto)

    # Comprobamos si mostrar las opciones de debug
    if args[cons.SHOW_DEBUG_OUTPUTS]:
        cons.loggin_level = logging._levelToName[logging.DEBUG]
    #else: logging._levelToName(logging.INFO)

    # Ponemos
    sizes = args[cons.INPUT_SIZE].split(cons.SEPARADOR_INPUT_IMAGENES)    
    args[cons.ALTURA_IMG] = int(sizes[0])
    args[cons.ANCHURA_IMG] = int(sizes[1] if len(sizes) > 1 else sizes[0])
    
def get_dict_args():    

    logger.debug("get_dict_args")
    args = vars(get_args_parser().parse_args())    
    logger.debug("check_args")
    check_args(args)
    
    logger.info("Arguments:" + 
                '\n'.join([(f"\t{k}: {args[k]}") for k in args]) +
                '\n' + ('-' * cons.NUM_GUIONES)
                )
    return args


#test
if __name__ == "__main__":
    dict_params = get_dict_args()

    logger.debug(dict_params)
    #logger.debug('-' * cons.NUM_GUIONES)

    for k in dict_params:
        logger.debug(f"dict_params[{k}]: {dict_params[k]}")
