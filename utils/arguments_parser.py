import argparse
from . import constants as con
import logging

def get_args_parser():
    parser = argparse.ArgumentParser(prog="Proyecto", 
                                     description="Modelo")
    
    parser.add_argument(f"--{con.NAME_BATCH_SIZE}", default=64, type=int,
                    help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    
    parser.add_argument(f"--{con.NAME_EPOCS}", default=50, type=int)

    parser.add_argument(f"--{con.NAME_INPUT_SIZE}", default=224, type=int,
                    help="images input size")
    
    parser.add_argument(f"--{con.NAME_DROP_PATH}", type=float, default=0.1,
                        help="Drop path rate (default: 0.1)")
    
    parser.add_argument(f"--{con.NAME_WEIGTH_DECAY}", type=float, default=0.05,
                        help="weight decay (default: 0.05)")
    
    parser.add_argument(f"--{con.NAME_LEARNING_RATE}", type=float, default=None,
                        help="learning rate (absolute lr)")
    
    parser.add_argument(f"--{con.NAME_BASE_LEARNING_RATE}", type=float, default=1e-3,
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    
    parser.add_argument(f"--{con.NAME_WARMUP_EPOCHS}", type=int, default=10, help="epochs to warmup LR")
        
    parser.add_argument(f"--{con.NAME_NUMBER_CLASSES}", default=5, type=int,
                        help="number of the classification types")

    parser.add_argument(f"--{con.NAME_DATA_PATH}", default="../Datasets/MESSIDOR2", type=str,
                        help="dataset path")
    parser.add_argument(f"--{con.NAME_OUTPUT_DIR}", default="./output_dir", type=str,
                        help="path where to save, empty for no saving")
    
    parser.add_argument(f"--{con.NAME_DEVICE}", default="cuda", type=str,  
                        help="device to use for training / testing")
    
    parser.add_argument(f"--{con.NAME_MODEL}", default="vgg19", type=str,
                        help="The model's name")
                        #help="Path to the local model or the model's name")
    
    parser.add_argument(f"--{con.PARTIAL_MODELS_PATH}", default="./partial_models", type=str,
                    help="Path to the folder where the best models will be stored")
    
    parser.add_argument(f"--{con.SUMMARIES_PATH}", default="./Summaries", type=str,
                help="Path to the folder where the summaries will be stored")

    parser.add_argument(f"--{con.NAME_MODEL_WEIGHTS}", default=None, type=str,
                help="Model's weights path")


    return parser

def check_args(args:dict):

    if args[con.NAME_MODEL] not in con.POSSIBLE_MODELS:
        texto = f"{con.NAME_MODEL}. Received: {args[con.NAME_MODEL]}. Expected one of {str(con.POSSIBLE_MODELS)}"
        logging.error(texto)
        argparse.ArgumentError(None, texto)

    
def get_dict_args():    

    logging.debug("get_dict_args")
    args = vars(get_args_parser().parse_args())    
    logging.debug("check_args")
    check_args(args)
    
    logging.info("Arguments:")
    for k in args:
        logging.info(f"\t{k}: {args[k]}")
    logging.info('-' * con.NUM_GUIONES)

    return args


#test
if __name__ == "__main__":
    dict_params = get_dict_args()

    logging.debug(dict_params)
    logging.debug('-' * con.NUM_GUIONES)

    for k in dict_params:
        logging.debug(f"dict_params[{k}]: {dict_params[k]}")