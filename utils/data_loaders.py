import os
from typing import *

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DistributedSampler

import utils.constants as cons
import torch

# Esto es para tener el logger
from utils.log_writer import getLogWritter
logger = getLogWritter(__name__)

def load_datasets(args:dict, is_main_device) -> dict:

    transform = transforms.Compose([
        # TODO: Para hacer el aumento de datos MODIFICAR AQUÍ ==>
        transforms.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),
        transforms.ToTensor()
    ])

    if args[cons.SHOW_DEBUG_OUTPUTS] and is_main_device:
        logger.debug("load_datasets")
        dict_datasets = dict()

        for folder_name in cons.LIST_FOLDER_NAMES:
            if is_main_device:
                logger.debug(f"folder name: {folder_name}. Data path: {args[cons.DATA_PATH]}.")
            path = os.path.join(args[cons.DATA_PATH], folder_name)
            if is_main_device:
                logger.debug(f"path: {path}")
            dict_datasets[folder_name] = datasets.ImageFolder(path, transform = transform)             

        for data_set in dict_datasets:
            logger.debug(f"{data_set}:\n\t{dict_datasets[data_set]}")

            for i in range(3):
                img, label = dict_datasets[data_set][i]
                logger.debug(f"Sample {i} from {folder_name} - img shape: {img.shape}, label: {label}, img min: {img.min()}, img max: {img.max()}")
   

        return dict_datasets
    else:
        return {
            folder_name:
            datasets.ImageFolder(root = os.path.join(args[cons.DATA_PATH], folder_name),
                                transform = transform) 
            for folder_name in cons.LIST_FOLDER_NAMES
        }    

def load_samplers(args:dict, datasets: Tuple)-> dict:    

    dict_samplers = {        
        #folder_name: torch.utils.data.SequentialSampler(folder_name)
        folder_name:
            ( DistributedSampler(datasets[cons.TRAIN_FOLDER_NAME], shuffle=True, num_replicas=os.environ["WORLD_SIZE"], rank=os.environ["LOCAL_RANK"]) if args[cons.IS_DISTRIBUTED] else
                torch.utils.data.SequentialSampler(datasets[folder_name]) )
        for folder_name in cons.LIST_FOLDER_NAMES
    }
    
    #if args[cons.IS_DISTRIBUTED]: dict_samplers[cons.TRAIN_FOLDER_NAME] = DistributedSampler(datasets[cons.TRAIN_FOLDER_NAME], shuffle=True)

    return dict_samplers

# TODO: Por comprobar de que está bien
def load_data_loaders(args: dict, datasets:dict) -> dict:

    samplers = load_samplers(args=args, datasets=datasets)

    return {
        folder_name : DataLoader(
            datasets[folder_name], 
            sampler=samplers[folder_name],
            batch_size=args[cons.BATCH_SIZE],
            drop_last=False,
            pin_memory=False,
            
            # shuffle = True # Con el Sampler debería hacerse automáticamente.
        )
        for folder_name in cons.LIST_FOLDER_NAMES
    }
