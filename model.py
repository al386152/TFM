import torch
import torch.nn as nn

def congelar_capas_desde_final(model, capas_entrenar_final = -1):

    if capas_entrenar_final == -1:
        return model
    # else: congela las capas

    print(f"Preparando congelar todas las capas menos las {capas_entrenar_final} últimas para el transfer learning")    
    # Esto es para congelar las capas
    list_model_children = list()
    
    print("for block in model.blocks:")
    for block in model.blocks:
        print(block)
        for child in block.children():
            print(f"{child} |",end=' ')    
            list_model_children.append(child)
            print("len(list(child.parameters())): ", len(list(child.parameters())))            
    print('-' * 4)
    
    print("len(list_model_children): ", len(list_model_children))
    if capas_entrenar_final != 0:
        list_model_children = list_model_children[: -1 * capas_entrenar_final]
    # else: list_model_children = list_model_children
    print("len(list_model_children): ", len(list_model_children))    

    for child in list_model_children:
        print(f"{child} |",end=' ')
        for param in child.parameters():
            #print(f"{param} |\|",end=' ')
            param.requires_grad = False   

    # Añadimos nuevas capas
    #num_ftrs = model.fc.in_features    
    #num_out = model.fc.out_features
    #print("model.fc.out_features: ", model.fc.out_features)
    #model.fc = nn.Linear(num_ftrs, salidas)
    
    # Nota: En algún punto, ya se añade una capa con las salidas esperadas.
    print(f"{'-'*4} Fin función transfer learning {'-'*4}")

    return model


def transfer_learning(model, capas_entrenar_final = -1, salidas = 5):

    model = congelar_capas_desde_final(model, capas_entrenar_final)
    num_ftrs = model.fc.in_features    
    #num_out = model.fc.out_features
    print("model.fc.out_features: ", model.fc.out_features)
    model.fc = nn.Sequential(
                nn.Linear(num_ftrs, salidas),
                nn.LayerNorm(salidas)
            )
    

    return model