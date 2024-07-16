import os
from typing import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DistributedSampler

import utils.constants as cons
import torch

# Esto es para tener el logger
from utils.log_writer import getLogWritter
logger = getLogWritter(__name__)

def load_datasets(args:dict, is_main_device) -> dict:

    # Transformaciones de las imagens de Validación y Prueba
    transform = v2.Compose([
        v2.PILToTensor(),        
        v2.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),
        #v2.ToTensor() # deprecated
        # The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.Output is equivalent up to float precision.
        v2.ToDtype(torch.float32, scale=True), 
        #v2.Normalize() # TODO: Ver si añado normalización
    ])  
        

    # Transformaciones de las imagens de entrenamiento
    training_transform = v2.Compose([        
        v2.PILToTensor(),
        
        v2.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),
        v2.RandomHorizontalFlip(), # Probabilidad de 0.5 (no me parece nesario poner una variable)
        v2.RandomRotation(cons.ROTATION_DEGREES),
        v2.ColorJitter(brightness=args[cons.COLOR_JITTER_BRIGHTNESS], contrast=args[cons.COLOR_JITTER_CONTRAST], 
                       saturation=args[cons.COLOR_JITTER_SATURATION], hue=args[cons.COLOR_JITTER_HUE]),         
        # TODO: Para hacer el aumento de datos MODIFICAR AQUÍ ==>
        v2.ToDtype(torch.float32, scale=True),
    ])

    if not args[cons.SHOW_DEBUG_OUTPUTS]:
        return {
            folder_name:
            datasets.ImageFolder(root = os.path.join(args[cons.DATA_PATH], folder_name),
                                transform = transform if folder_name != cons.TRAIN_FOLDER_NAME else training_transform) 
            for folder_name in cons.LIST_FOLDER_NAMES
        }    
    else:
        # La versión de debug:
        if is_main_device:
            logger.debug("load_datasets")
        dict_datasets = dict()

        for folder_name in cons.LIST_FOLDER_NAMES:
            if is_main_device:
                logger.debug(f"folder name: \"{folder_name}\". Data path: \"{args[cons.DATA_PATH]}\".")
            path = os.path.join(args[cons.DATA_PATH], folder_name)
            if is_main_device:
                logger.debug(f"path: {path}")
            dict_datasets[folder_name] = datasets.ImageFolder(path, transform = transform if folder_name != cons.TRAIN_FOLDER_NAME else training_transform)           

        if is_main_device:
            for data_set in dict_datasets:
                logger.debug(f"{data_set}:\n\t{dict_datasets[data_set]}")

                for i in range(3):
                    img, label = dict_datasets[data_set][i]
                    logger.debug(f"Sample {i} from {data_set} - img shape: {img.shape}, label: {label}, img min: {img.min()}, img max: {img.max()}")
   

        return dict_datasets        

def load_samplers(args:dict, datasets: Tuple)-> dict:    
        
    world_size = 1 if ( "WORLD_SIZE" not in os.environ) else int(os.environ["WORLD_SIZE"])
    rank = 0 if ( "LOCAL_RANK" not in os.environ) else int(os.environ["LOCAL_RANK"])

    if rank == 0:
        logger.debug(f"world_size: {world_size} (type: {type(world_size)})")
        logger.debug(f"rank: {rank} (type: {type(rank)})")
        #logger.debug(f"args[cons.IS_DISTRIBUTED]: {args[cons.IS_DISTRIBUTED]}") 

    dict_samplers = {        
        #folder_name: torch.utils.data.SequentialSampler(folder_name)
        folder_name:
            ( DistributedSampler(datasets[folder_name], shuffle=True, num_replicas=world_size, rank=rank) if args[cons.IS_DISTRIBUTED] else
                torch.utils.data.SequentialSampler(datasets[folder_name]) )
        for folder_name in cons.LIST_FOLDER_NAMES
    }
    
    #if args[cons.IS_DISTRIBUTED]: dict_samplers[cons.TRAIN_FOLDER_NAME] = DistributedSampler(datasets[cons.TRAIN_FOLDER_NAME], shuffle=True)

    return dict_samplers

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
