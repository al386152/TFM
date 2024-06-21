import argparse
import constants as con


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
    parser.add_argument(f"--{con.NAME_OUTPUT_DIR}", default="./output_dir",
                        help="path where to save, empty for no saving")
    
    parser.add_argument(f"--{con.NAME_DEVICE}", default="cuda", 
                        help="device to use for training / testing")
    
    return parser

def get_dict_args():
    args = get_args_parser()
    return vars(args)


if __name__ == "__main__":

    #print(args)

    dict_params = get_dict_args()

    print(dict_params)
    print('-' * 8)

    for k in dict_params:
        print(f"dict_params[{k}]: {dict_params[k]}")