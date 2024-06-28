from typing import *
from datetime import datetime

import utils.arguments_parser as ap
import utils.constants as con
import utils.model_related as mr
import os
from datetime import datetime

import torch
from torchvision import datasets, transforms
from torchvision.models import vgg19, resnet50
from torch.utils.tensorboard import SummaryWriter


# TODO: Añadir de que, si se para como "nombre" la ruta a un modelo, se cargue ese.
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

def train_model(args: dict, model, samplers, device):
    
    print( ('-' * con.NUM_GUIONES) + "\nStarting to train the model\n" + ('-' * con.NUM_GUIONES) )

    # Preparando las variables

    model = model.to(device)
    best_vloss = float('inf') # Número imposible
    partial_models_path = args[con.PARTIAL_MODELS_PATH] 
    summaries_path = args[con.SUMMARIES_PATH]
    #save_model = False  

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

        print(f"Epoch [{epoch + 1}]:")

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
        print(f"Avg.loss: {avg_loss} | Avg.validation loss: {avg_val_loss}")
        
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
            #save_model = True
        else:
            #save_model = False
            pass

        # Para el early stopping
        if early_stopper.early_stop(running_val_loss):
            print(f"Stopping the training. Running validation loss: {running_val_loss}, 'patience': {early_stopper.patience}, min_diff: {early_stopper.min_delta}")
            break
        print('') # Para añadir un salto de línea
    
    print(('-' * con.NUM_GUIONES) + "\nTraining ended\n" + ('-' * con.NUM_GUIONES))
    #return #save_model

def saving_the_model(args: dict, model):
    model_name = con.OUTPUT_MODEL_NAME(name=args[con.NAME_MODEL], number_clases=args[con.NAME_NUMBER_CLASSES])
    print(f"Guardado el modelo con el nombre: {model_name}")
    torch.save(model.state_dict(),  model_name)

def inference(args: dict, model, samplers):
    pass

def get_training_time_HMS_format(time_start, time_end):
    intervalo = time_end - time_start

    days, seconds = intervalo.days, intervalo.seconds

    h = days * 24 + seconds // 3600
    m = (seconds % 3600) // 60
    s = (seconds % 60)

    return f"{h:02}:{m:02}:{s:02}"

def main(args: dict):

    # Estableciendo el dispositivo en el que se va a trabajar.    
    device = "cpu"
    if "cuda" in args[con.NAME_DEVICE] :
        if not torch.cuda.is_available():
            print(" 'torch.cuda' is not available ==> cpu")            
            #device = "cpu"
        else:
            device = args[con.NAME_DEVICE]
    #else: device = "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    datasets = load_datasets(args)
    #print(f"Datasets:\n{datasets}\n---")
    samplers = load_data_loaders(args, datasets)
    #print(f"Samplers:\n{samplers}\n---")
    model = load_model(args)
    print(f"Model: {model}")
    #save_model = train_model(args, model=model, samplers=samplers, device=device)
    t_inicio = datetime.now()
    train_model(args, model=model, samplers=samplers, device=device)
    tiempo_entrenamiento = get_training_time_HMS_format(t_inicio, datetime.now())
    print(f"Tiempo entrenamiento: {tiempo_entrenamiento}")
    #if save_model: saving_the_model(args, model)
    saving_the_model(args, model)

if __name__ == "__main__":    
    main(ap.get_dict_args())