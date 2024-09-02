import os
from typing import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DistributedSampler, random_split

import utils.constants as cons
import torch

# Esto es para tener el logger
from utils.log_writer import getLogWritter
logger = getLogWritter(__name__)

def load_datasets_one_path_three_folders(args:dict, is_main_device, transform:v2.Compose, training_transform: v2.Compose) -> dict:
    # Transformaciones de las imagens de entrenamiento

    if not args[cons.SHOW_DEBUG_OUTPUTS]:
        return {
            folder_name:
            datasets.ImageFolder(root = os.path.join(args[cons.DATA_PATH], folder_name),
                                transform = transform if folder_name != cons.TRAIN_FOLDER_NAME else training_transform) 
            for folder_name in cons.LIST_FOLDER_NAMES
        }    
    else:
        # La versión de debug:
        dict_datasets = dict()

        for folder_name in cons.LIST_FOLDER_NAMES:
            if is_main_device:
                logger.debug(f"folder name: \"{folder_name}\". Data path: \"{args[cons.DATA_PATH]}\".")

            path = os.path.join(args[cons.DATA_PATH], folder_name)
            if is_main_device:
                logger.debug(f"path: {path}")

            dict_datasets[folder_name] = datasets.ImageFolder(path, transform = transform if folder_name != cons.TRAIN_FOLDER_NAME else training_transform)               

        return dict_datasets

def load_datasets_one_path_one_folder(args:dict, is_main_device, transform:v2.Compose, training_transform: v2.Compose) -> dict:

    dataset_sizes = args[cons.SPLIT_PERCENTAGES]

    full_dataset = datasets.ImageFolder(root = args[cons.DATA_PATH], transform = transform) 

    if is_main_device: 
        logger.debug(f"full_dataset:\n{full_dataset}")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, dataset_sizes)    

    if is_main_device: 
        logger.debug(f"train_dataset:\n{train_dataset}")
        logger.debug(f"val_dataset:\n{val_dataset}")
        logger.debug(f"test_dataset:\n{test_dataset}")    

    train_dataset.transform = training_transform    

    dict_datasets = {
        cons.TEST_FOLDER_NAME: train_dataset, 
        cons.VALIDATION_FOLDER_NAME: val_dataset, 
        cons.TRAIN_FOLDER_NAME: test_dataset
    }

    return dict_datasets

def load_dataset_three_paths(args:dict, is_main_device, transform:v2.Compose, training_transform: v2.Compose) -> dict:
    

    dict_datasets = {
        cons.TEST_FOLDER_NAME: datasets.ImageFolder(root = args[cons.TEST_DATA_PATH], transform = transform), 
        cons.VALIDATION_FOLDER_NAME: datasets.ImageFolder(root = args[cons.VALIDATION_DATA_PATH], transform = transform), 
        cons.TRAIN_FOLDER_NAME: datasets.ImageFolder(root = args[cons.TRAIN_DATA_PATH], transform = training_transform)
    }

    if is_main_device:
        logger.debug(f"test_dataset:\n{dict_datasets[cons.TEST_FOLDER_NAME]}")
        logger.debug(f"val_dataset:\n{dict_datasets[cons.VALIDATION_FOLDER_NAME]}")
        logger.debug(f"train_dataset:\n{dict_datasets[cons.TRAIN_FOLDER_NAME]}")    
    
    return dict_datasets

def load_dataset_two_paths(args:dict, is_main_device, transform:v2.Compose, training_transform: v2.Compose) -> dict:

    dataset_sizes = args[cons.SPLIT_PERCENTAGES]
    
    train_val_dataset = datasets.ImageFolder(root = args[cons.DATA_PATH], transform = transform)
    test_dataset = datasets.ImageFolder(root = args[cons.TEST_DATA_PATH], transform = transform)

    if is_main_device: 
        logger.debug(f"train_val_dataset:\n{train_val_dataset}")
        logger.debug(f"test_dataset:\n{test_dataset}")

    train_dataset, val_dataset = random_split(train_val_dataset, dataset_sizes)    

    if is_main_device: 
        logger.debug(f"train_dataset:\n{train_dataset}")
        logger.debug(f"val_dataset:\n{val_dataset}")

    train_dataset.transform = training_transform    

    dict_datasets = {
        cons.TEST_FOLDER_NAME: train_dataset, 
        cons.VALIDATION_FOLDER_NAME: val_dataset, 
        cons.TRAIN_FOLDER_NAME: test_dataset
    }

    return dict_datasets

def get_validation_test_transform(args):
    return v2.Compose([
        v2.PILToTensor(),        
        v2.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),
        v2.ToDtype(torch.float32, scale=True), 
    ])

def get_training_transform(args):
    # Nota: si añades alguna más, métela también en "optuna_related" para que haga la prueba.
    return v2.Compose([        
        v2.PILToTensor(),
        v2.Resize((args[cons.ALTURA_IMG], args[cons.ANCHURA_IMG])),        
        v2.RandomHorizontalFlip(cons.P_H_FLIP), 
        v2.RandomVerticalFlip(cons.P_V_FLIP), 
        v2.RandomPerspective(distortion_scale=cons.RANDOM_PERSPECTIVE_DISTORSION_SCALE), # Probabilidad de 0.5
        v2.RandomRotation(cons.ROTATION_DEGREES),
        v2.ColorJitter(brightness=args[cons.COLOR_JITTER_BRIGHTNESS], contrast=args[cons.COLOR_JITTER_CONTRAST], 
                       saturation=args[cons.COLOR_JITTER_SATURATION], hue=args[cons.COLOR_JITTER_HUE]),         
        # NOTA: Para hacer el aumento de datos MODIFICAR AQUÍ ==>
        v2.ToDtype(torch.float32, scale=True),
    ])

def load_datasets(args:dict, is_main_device) -> dict:

    if is_main_device:
        logger.debug("load_datasets")

    # Transformaciones de las imagens de Validación y Prueba
    transform = get_validation_test_transform(args)

    if is_main_device:
        logger.debug(f"transform created:\n{transform}")

    training_transform = get_training_transform(args)    
    
    if is_main_device:
        logger.debug(f"transform created:\n{training_transform}")

    # Comprobación del tipo de función que hay que usar        
    if args[cons.DATA_PATH] != None:
        
        if args[cons.TEST_DATA_PATH] != None:
            # 2 carpetas, una de validación y entrenamiento y otra de test.
            func_load_dataset = load_dataset_two_paths
        else: 
            # Diferencia entre sacar todo de 1 carpeta o de 3 carpetas: que en las carpetas existan test, val y train:        
            #   Voy a asumir que si existe una de ellas, existen todas (si no, saltará error y será tema del usuario).
            if os.path.isdir(os.path.join(args[cons.DATA_PATH], cons.TEST_FOLDER_NAME)):
                func_load_dataset = load_datasets_one_path_three_folders
            else:
                func_load_dataset = load_datasets_one_path_one_folder        
    else:        
        func_load_dataset = load_dataset_three_paths
        

    dict_datasets = func_load_dataset(args, is_main_device, transform, training_transform)

    if args[cons.SHOW_DEBUG_OUTPUTS] and is_main_device:
            for data_set in dict_datasets:
                logger.debug(f"{data_set}:\n\t{dict_datasets[data_set]}")

                for i in range(3):  # "3" para tener unos pocos ejemplos, podría ser 1 podrían ser todos.
                    img, label = dict_datasets[data_set][i]
                    logger.debug(f"Sample {i} from {data_set} - img shape: {img.shape}, label: {label}, img min: {img.min()}, img max: {img.max()}")
   

    return dict_datasets
            
def load_samplers(args:dict, datasets: Tuple)-> dict:    
        
    world_size = 1 if ( "WORLD_SIZE" not in os.environ) else int(os.environ["WORLD_SIZE"])
    rank = 0 if ( "LOCAL_RANK" not in os.environ) else int(os.environ["LOCAL_RANK"])

    if rank == 0:
        logger.debug(f"world_size: {world_size} (type: {type(world_size)})")
        logger.debug(f"rank: {rank} (type: {type(rank)})")

    dict_samplers = {        
        folder_name:
            ( DistributedSampler(datasets[folder_name], shuffle=True, num_replicas=world_size, rank=rank) if args[cons.IS_DISTRIBUTED] else
                torch.utils.data.SequentialSampler(datasets[folder_name]) )
        for folder_name in cons.LIST_FOLDER_NAMES
    }

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
