import sys
from os import environ

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed
from torchvision.models import vgg19, resnet50, resnet152, densenet121, densenet169

SWITCH_MODELOS = {
    "vgg19": vgg19,
    "resnet50": resnet50,
    "resnet152": resnet152,
    "densenet121": densenet121,
    "densenet169": densenet169
}

def get_classifier_layer(model: torch.nn.Module, model_name: str, is_main_device: bool=False)->torch.nn.modules.linear.Linear:

    if "vgg" in model_name:
        classifier_layer = model.classifier[6]        
    #elif model_name == "resnet50":
    elif "resnet" in model_name:
        classifier_layer = model.fc
    elif "densenet" in model_name:
        classifier_layer = model.classifier
    else:        
        #if is_main_device: 
        #    logger.warning(f"Han habido varias comprobaciones antes, ¿cómo has llegado aquí?. 'model_name: {model_name}'")
        classifier_layer = None

    return classifier_layer
# -- Fin get_classifier_layer -- #   

def poner_outputs_bien(model:torch.nn.Module, model_name:str, outputs = 5):

    classifier_layer = get_classifier_layer(model, model_name)
    num_ftrs = classifier_layer.in_features

    capa_clasificacion = torch.nn.Linear(num_ftrs, outputs)
    if "vgg" in model_name:
        model.classifier[6] = capa_clasificacion
    elif "resnet" in model_name:
        model.fc = capa_clasificacion
    elif "densenet" in model_name:
        model.classifier = capa_clasificacion
    
    return model
# -- FIN modify_model_layers -- #

def main(argv):
    modelo = argv[1]    
    pesos = argv[2]
    pesos_bien = argv[3]
    
    device = environ["LOCAL_RANK"]

    print(device)
    #print(f'environ["CUDA_VISIBLE_DEVICES"]: {environ["CUDA_VISIBLE_DEVICES"]}')
    #map_location = torch.cuda.set_device(int(environ["LOCAL_RANK"])) if torch.cuda.is_available() else torch.device('cpu')
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    if device == "0":
        print(f"modelo: {modelo}")
        print(f"pesos: {pesos}")    
        print(f"pesos_bien: {pesos_bien}")

        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")

        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"map_location: {map_location}")


    distributed.init_process_group(backend="nccl")
    model = poner_outputs_bien(SWITCH_MODELOS[modelo](), modelo)
    model = model.to(map_location)
    model = DistributedDataParallel(model, device_ids=[map_location])    
    model.load_state_dict(torch.load(pesos, map_location=map_location), strict=True)
    torch.save(model.module.state_dict(), pesos_bien)


# argv[1] modelo
# argv[2] path pesos DistributedDataParallel
# argv[3] path pesos Bien hechos.

if __name__ == "__main__":
    #environ["LOCAL_RANK"] = "0"
    #environ["RANK"] = "0"
    #environ["WORLD_SIZE"] = "1"
    argv = sys.argv
    #print(f'environ["LOCAL_RANK"]: {environ["LOCAL_RANK"]}')    
    #print(f'environ["RANK"]: {environ["RANK"]}')
    if environ["LOCAL_RANK"] == "0" or environ["RANK"] == "0":
        print(f"argv: {argv}")
    main(argv)
