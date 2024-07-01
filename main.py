import os

from typing import *
from datetime import datetime
import logging

import torch.utils
import torch
from torchvision import datasets, transforms

import utils.arguments_parser as ap
import utils.constants as con
import utils.model_related as mr

from torch.utils.tensorboard import SummaryWriter


def load_model(args: dict, fine_tuning:bool = True, capas_entrenar_final = -1) -> torch.nn.Module:

    model = None
    model_name = args[con.NAME_MODEL]
    model_weights_path = args[con.NAME_MODEL_WEIGHTS]

    logging.debug(f"model_weights_path: {model_weights_path}")

    if model_weights_path is not None and os.path.isfile(model_weights_path):        
        model_weights_path = model_weights_path
        weights = None
    else: 
        weights = con.DEFAULT_MODEL_WEIGHTS

    logging.info(f"Cargando los siguientes pesos: \'{weights if weights is not None else model_weights_path}\'")

    model = con.SWITCH_MODELOS[model_name](weights = weights)

    if fine_tuning: 
        model = mr.fine_tuning(model=model, model_name=model_name, 
                    outputs=args[con.NAME_NUMBER_CLASSES])
                
    if model_weights_path:
        # Es importante cargarlo *DESPUES* del fine tuning
        model.load_state_dict(torch.load(model_weights_path))        
    
    model = mr.transfer_learning(model, capas_entrenar_final)
        
    return model

def load_datasets(args:dict) -> dict:

    transform = transforms.Compose([
        # Aquí se puede añadir el aumento de datos, aunque prefiero que estén guardados en ficheros.
        transforms.Resize((con.HEIGHT_IMAGES, con.WIDTH_IMAGES)),
        transforms.ToTensor()
    ])

    return {
        folder_name:
        datasets.ImageFolder(root = os.path.join(args[con.NAME_DATA_PATH], folder_name),
                             transform = transform) 
        for folder_name in con.LIST_FOLDER_NAMES
    }

def load_samplers(datasets: Tuple)-> dict:

    return {        
        #folder_name: torch.utils.data.SequentialSampler(folder_name)
        folder_name: torch.utils.data.RandomSampler(datasets[folder_name])
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

def train_model(args: dict, model, samplers, device):
    
    logging.info( ('-' * con.NUM_GUIONES) + "\nStarting to train the model\n" + ('-' * con.NUM_GUIONES) )

    # Preparando las variables

    model = model.to(device)
    best_vloss = float('inf') # Número imposible para que en la primera iteración sea menor sí o sí.
    partial_models_path = args[con.PARTIAL_MODELS_PATH] 
    summaries_path = args[con.SUMMARIES_PATH]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter( os.path.join(partial_models_path, f'Proyecto_{timestamp}'))
    loss_fn = torch.nn.CrossEntropyLoss()
    early_stopper = mr.EarlyStopper(patience=3, min_delta=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Creando carpetas para las salidas  
    if not os.path.isdir(partial_models_path):
        os.makedirs(partial_models_path)
    if not os.path.isdir(summaries_path):
        os.makedirs(summaries_path)

    # Entrenando
    for epoch in range(args[con.NAME_EPOCS]):

        logging.info(f"Epoch [{epoch + 1}]:")

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

                logging.debug("vinputs:\n", vinputs)
                logging.debug("vlabels:\n", vlabels)
                logging.debug("voutputs:\n", voutputs)
    
        avg_val_loss = running_val_loss / (i + 1)
        logging.info(f"Avg.loss: {avg_loss} | Avg.validation loss: {avg_val_loss}")
        
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_val_loss },
                        epoch + 1)
        writer.flush()

        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_name = f"model_{epoch}_{timestamp}.pth"
            model_path = os.path.join(args[con.PARTIAL_MODELS_PATH], 
                                      model_name) 

            torch.save(model.state_dict(), model_path)

        # Para el early stopping
        if early_stopper.early_stop(running_val_loss):
            logging.info(f"Stopping the training. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")
            break        
    
    logging.info('\n' + ('-' * con.NUM_GUIONES) + "\nTraining ended\n" + ('-' * con.NUM_GUIONES))

def saving_the_model(args: dict, model):
    model_name = con.OUTPUT_MODEL_NAME(name=args[con.NAME_MODEL], number_clases=args[con.NAME_NUMBER_CLASSES])
    logging.info(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)

def inference(args: dict, model, samplers):
    model.eval()

    
def get_training_time_HMS_format(time_start, time_end):
    intervalo = time_end - time_start

    days, seconds = intervalo.days, intervalo.seconds

    h = days * 24 + seconds // 3600
    m = (seconds % 3600) // 60
    s = (seconds % 60)

    return f"{h:02}:{m:02}:{s:02}"

def main(args: dict):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


    # Estableciendo el dispositivo en el que se va a trabajar.    
    device = "cpu"
    if "cuda" in args[con.NAME_DEVICE]:
        logging.debug(f"cuda in {args[con.NAME_DEVICE]}")
        if not torch.cuda.is_available():
            logging.info(" 'torch.cuda' is not available ==> cpu")            
            #device = "cpu"
        else:
            device = args[con.NAME_DEVICE]
    #else: device = "cpu"

    device = torch.device(device)
    logging.info(f"Device: {device}")

    datasets = load_datasets(args)
    
    logging.debug(
        "\n".join([f"len(dataset): {len(dataset)}\ndataset:\n{str(dataset)}" for dataset in datasets])
    )

    logging.debug(f"Datasets:\n{datasets}\n---")
    samplers = load_data_loaders(args, datasets)
    
    logging.debug(
        "\n".join([f"len(sampler): {len(sampler)}\nsampler:\n{str(sampler)}" for sampler in samplers])
    )

    
    logging.debug(f"Samplers:\n{samplers}\n---")

    model = load_model(args)
    logging.info(f"Model: {model}")
    #save_model = train_model(args, model=model, samplers=samplers, device=device)
    t_inicio = datetime.now()
    train_model(args, model=model, samplers=samplers, device=device)
    tiempo_entrenamiento = get_training_time_HMS_format(t_inicio, datetime.now())
    logging.info(f"Tiempo entrenamiento: {tiempo_entrenamiento}")
        
    metricas = mr.evaluate_model(model=model,dataloader=samplers[con.VALIDATION_FOLDER_NAME], device=device)
    #accuracy, roc_auc, pr_auc, f1, conf_matrix = metricas
    saving_the_model(args, model)

if __name__ == "__main__":
    main(ap.get_dict_args())