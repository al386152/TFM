from typing import *

import utils.arguments_parser as ap
import utils.constants as con
import utils.model_related as mr
import os
from datetime import datetime

import torch
from torchvision import datasets
from torchvision.models import vgg19, resnet50
from torch.utils.tensorboard import SummaryWriter


# TODO: comprobar de que vaya bien
def load_model(args: dict, fine_tuning:bool = True, capas_entrenar_final = -1) -> torch.nn.Module:

    model = None
    if args[con.NAME_MODEL] == "vgg19":
        model = vgg19(weights="IMAGENET1K_V1")
    elif args[con.NAME_MODEL] == "resnet50":
        model = resnet50(weights="IMAGENET1K_V1")
    #else: pass

    if fine_tuning: 
        model = mr.fine_tuning(model=model, model_name=args[con.NAME_MODEL], 
                    outputs=args[con.NAME_NUMBER_CLASSES])
    
    model = mr.transfer_learning(model, capas_entrenar_final)
        
    return model

# TODO: Por comprobar de que está bien
def load_samplers(datasets: Tuple)-> dict:
    
    #dataset_train, dataset_val, dataset_test = datasets  
    
    #sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    #return (sampler_test, sampler_train, sampler_val)

    return {
        folder_name: torch.utils.data.SequentialSampler(folder_name)
        for folder_name in con.LIST_FOLDER_NAMES
    }

# TODO: Por comprobar de que está bien
def load_data_loaders(args: dict, datasets:dict) -> dict:

    samplers = load_samplers(datasets=datasets)

    return {
        folder_name : torch.utils.data.DataLoader(
            datasets[folder_name], sampler=samplers[folder_name],
            batch_size=args[con.NAME_BATCH_SIZE],
            # pin_memory=args.pin_mem,
            # drop_last=True,
            # shuffle = True # Con el Sampler debería hacerse automáticamente.
        )
        for folder_name in con.LIST_FOLDER_NAMES
    }

# TODO: Comprobar de que está bien.
def load_datasets(args:dict) -> dict:

    print()

    return {
        folder_name:
        datasets.ImageFolder(root = os.path.join(args[con.NAME_DATA_PATH], folder_name))
        for folder_name in con.LIST_FOLDER_NAMES
    }

    #train_dataset = datasets.ImageFolder(root = args[os.path.join(con.NAME_DATA_PATH, con.TEST_FOLDER_NAME)])
    #val_dataset = datasets.ImageFolder(root = args[os.path.join(con.NAME_DATA_PATH, con.VALIDATION_FOLDER_NAME)])
    #test_dataset = datasets.ImageFolder(root = args[os.path.join(con.NAME_DATA_PATH, con.TRAIN_FOLDER_NAME)])
    #return  (train_dataset, val_dataset, test_dataset)

def train_model(args: dict, model, samplers, device):
    
    print( ("-"*8) + "\nStarting to train the model\n" + ("-"*8) )
    model = model.to(device)    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'Proyecto_{timestamp}')
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = mr.EarlyStopper(patience=3, min_delta=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    best_vloss = float('inf') # Número imposible

    for epoch in range(args[con.NAME_EPOCS]):

        print(f"Epoch {epoch + 1}:")

        model.train(True)
        avg_loss = mr.train_one_epoch(model=model, epoch_index=epoch, tb_writer=writer, 
                                      training_loader=samplers[con.TRAIN_FOLDER_NAME], 
                                      loss_func=loss_fn, optimizer=optimizer)
        running_val_loss = 0.0

        model.eval()
            # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(samplers[con.VALIDATION_FOLDER_NAME]):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_val_loss += vloss
    
        avg_val_loss = running_val_loss / (i + 1)
        print(f"Loss train {avg_loss} valid {avg_val_loss}")
        
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_val_loss },
                        epoch + 1)
        writer.flush()

        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        # Para el early stopping
        if early_stopper.early_stop(running_val_loss):
            print(f"Stopping the traning. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")
            break

def saving_the_model(args: dict, model):

    torch.save(model.state_dict(), con.OUTPUT_MODEL_NAME(name=args[con.NAME_MODEL], number_clases=args[con.NAME_NUMBER_CLASSES] ))

def inference(args: dict, model, samplers):
    pass


def main(args: dict):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = load_datasets(args)
    print(f"Datasets:\n{datasets}\n---")
    samplers = load_data_loaders(args, datasets)
    print(f"Samplers:\n{samplers}\n---")
    model = load_model(args)
    print(f"Model: {model}")
    train_model(args, model=model, samplers=samplers, device=device)


if __name__ == "__main__":    
    main(ap.get_dict_args())