from typing import *

import utils.arguments_parser as ap
import utils.constants as con

import torch

def load_model(args: dict) -> torch.nn.Module:

    # TODO: Por hacer entero
    model = object()

    return model

# TODO: Por comprobar de que está bien
def load_samplers(datasets: Tuple)-> Tuple[Iterable, Iterable, Iterable]:
    
    dataset_train, dataset_val, dataset_test = datasets  
    
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    return (sampler_test, sampler_train, sampler_val)

# TODO: Por comprobar de que está bien
def load_data_loaders(args: dict, datasets:Tuple) -> List[Iterable, Iterable, Iterable]:

    samplers = load_samplers(datasets=datasets)

    list_data_loaders = [
        torch.utils.data.DataLoader(
            datasets[i], sampler=samplers[i],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            # drop_last=True,
            # shuffle = True # TODO: Recuerda lo he añadido yo
        )
        for i in range(len(samplers))
    ]       

    return list_data_loaders

# TODO: Por hacer
def load_datasets():
    pass

def main(args: dict):
    
    datasets = load_datasets()
    samplers = load_data_loaders()
    model = load_model()



if __name__ == "__main__":    
    main(ap.get_dict_args())