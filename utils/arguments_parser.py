import argparse
import logging

from . import constants as cons
from .log_writer import getLogWritter

logger = getLogWritter(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser(prog="Proyecto", 
                                     description="Modelo")
    
    parser.add_argument(f"--{cons.NAME_BATCH_SIZE}", default=64, type=int,
                    help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    
    parser.add_argument(f"--{cons.NAME_EPOCS}", default=50, type=int)

    parser.add_argument(f"--{cons.NAME_INPUT_SIZE}", default=224, type=int,
                    help="images input size")
    
    parser.add_argument(f"--{cons.NAME_DROP_PATH}", type=float, default=0.1,
                        help="Drop path rate (default: 0.1)")
    
    parser.add_argument(f"--{cons.NAME_WEIGTH_DECAY}", type=float, default=0.05,
                        help="weight decay (default: 0.05)")
    
    parser.add_argument(f"--{cons.NAME_LEARNING_RATE}", type=float, default=None,
                        help="learning rate (absolute lr)")
    
    parser.add_argument(f"--{cons.NAME_BASE_LEARNING_RATE}", type=float, default=1e-3,
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    
    parser.add_argument(f"--{cons.NAME_WARMUP_EPOCHS}", type=int, default=10, help="epochs to warmup LR")
        
    parser.add_argument(f"--{cons.NAME_NUMBER_CLASSES}", default=5, type=int,
                        help="number of the classification types")

    parser.add_argument(f"--{cons.NAME_DATA_PATH}", default="../Datasets/MESSIDOR2", type=str,
                        help="dataset path")
    parser.add_argument(f"--{cons.NAME_OUTPUT_DIR}", default="./output_dir", type=str,
                        help="path where to save, empty for no saving")
    
    parser.add_argument(f"--{cons.NAME_DEVICE}", default="cuda", type=str,  
                        help="device to use for training / testing")
    
    parser.add_argument(f"--{cons.NAME_MODEL}", default="vgg19", type=str,
                        help="The model's name")
                        #help="Path to the local model or the model's name")
    
    parser.add_argument(f"--{cons.PARTIAL_MODELS_PATH}", default="./partial_models", type=str,
                    help="Path to the folder where the best models will be stored")    

    parser.add_argument(f"--{cons.NAME_MODEL_WEIGHTS}", default=None, type=str,
                help="Model's weights path")
    
    parser.add_argument(f"--{cons.NAME_SHOW_DEBUG_OUTPUTS}", default=False, type=bool,
            help="Show the debug outputs")
    

    return parser

def check_args(args:dict):

    if args[cons.NAME_MODEL] not in cons.POSSIBLE_MODELS:
        texto = f"{cons.NAME_MODEL}. Received: {args[cons.NAME_MODEL]}. Expected one of {str(cons.POSSIBLE_MODELS)}"
        logger.error(texto)
        argparse.ArgumentError(None, texto)
    
    if args[cons.NAME_SHOW_DEBUG_OUTPUTS]:
        cons.loggin_level = logging._levelToName(logging.DEBUG)
    #else: logging._levelToName(logging.INFO)
    
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