import os
import argparse
from typing import *
import shutil

import csv
import random

# Para la data augmentation
import cv2

random.seed(10)

TEST_FOLDER_PATH = "test"
TRAIN_FOLDER_PATH = "train"
VAL_FOLDER_PATH = "val"

def get_args_parser():
    parser = argparse.ArgumentParser('División de imagenes de forma aleatoria partir de CSV', add_help=False)    

    parser.add_argument("--csv_path", default="./", type=str, help="csv path")

    parser.add_argument("--data_path", default="./", type=str, help="data path")

    parser.add_argument("--output_path", default="./", type=str, help="output path")

    parser.add_argument("--train_test_val", default="80,10,10", type=str,
                        help="porcentajes de entrenamiento. Formato: train,test,val (ejemplo: 80,10,10)")

    parser.add_argument("--augment_data", default=False, type=bool,
                        help="True: Se realiza un aumento de datos para que todas las clases tengan el mismo número de elementos.")

    return parser

def leer_csv(csv_path: str, data_path: str) -> Dict[str, str]:

    dict_data = dict()
    with open(csv_path, newline='') as csvfile:
        #reader = csv.DictReader(csvfile=csvfile,fieldnames=None)
        reader = csv.reader(csvfile=csvfile, delimiter=' ')
        iterread = iter(reader) # Para saltarnos la primera línea
        next(iterread)
        for row in iterread:            
            nombre_fichero, tipo_datos = row[0], row[1]
            
            if tipo_datos not in dict_data:
                dict_data[tipo_datos] = list()

            dict_data[tipo_datos].append(os.path.join(data_path, nombre_fichero))
    
    return dict_data

def calcular_proporciones(datos: dict)-> Dict[str, float]:

    # Hay que mantener las proporciones entre clases
    num_total_elementos = 0  # al final será equivalente a: len(datos.values())
    dict_num_tipos = dict()
    #dict_porcentajes_tipos = dict()

    # Para poder calcular el porcentaje de elementos de cada clase
    for clave in datos:
        num = len(datos[clave])
        dict_num_tipos[clave] = num
        num_total_elementos += num
    
    #dict_porcentajes_tipos = 
    #return dict_porcentajes_tipos
    return {clave: (dict_num_tipos[clave] / num_total_elementos) for clave in dict_num_tipos}


def repartir_datos(datos: dict, dict_porcentajes_tipo: Dict[str, float]) -> Dict[str, Dict[str, str]]:

    dict_elementos_escogidos = dict()
    
    for tipo in dict_porcentajes_tipo:       
        for clase in datos:
            num_elements = dict_porcentajes_tipo[tipo] * len(datos[clase])

            #elementos_escogidos = random.sample(datos[clave], num_elements)
            dict_elementos_escogidos[tipo] = {clase: random.sample(datos[clase], num_elements)}
            # Eliminamos los elementos que ya hemos escogido para no volver a escogerlos
            datos[clase] = set(datos[clase]) - set(dict_elementos_escogidos[tipo][clase])
    
    return dict_elementos_escogidos

def copiar_y_generar_imagenes(output_path: str, dict_imagenes: dict):
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    dict_folder_paths = {
        TEST_FOLDER_PATH: os.path.join(output_path, TEST_FOLDER_PATH),
        TRAIN_FOLDER_PATH: os.path.join(output_path, TRAIN_FOLDER_PATH),
        VAL_FOLDER_PATH: os.path.join(output_path, VAL_FOLDER_PATH) 
    }

    for tipo_op in dict_folder_paths:

        if not os.path.isdir(dict_folder_paths[tipo_op]):
            os.makedirs(dict_folder_paths[tipo_op])
        print("Copiando los elementos para: ", tipo_op)

        for list_elems in dict_imagenes[tipo_op].values():
            for elem in list_elems:
                print("Copiando: ", elem)
                shutil.copy(elem, dict_folder_paths[tipo_op])
                print("Se ha copiado: ", elem)
                    

def data_augmentation(data: dict, tam_kernel = 7):        

    for image in data.vales():
        # Las imagenes siempre van a estar en las mismas 2 posiciones: ojo izquierdo y ojo derecho (imagino que, en general, la del derecho será como el del izquierdo pero con una rotación vertical)
        rotacion_vertical = cv2.flip(image, 1) # 1: vertical, 0: horizontal y -1: vertical y horizontal.
        # Entiendo que las fotos podrían salir borrosas
        ruido_gaussiano = cv2.blur( image, (tam_kernel, tam_kernel))
        # Las fotos podrían tener distintas iluminaciones (valores entre -255 y 255. Fuente: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
        iluminacion = cv2.convertScaleAbs(image, beta=random.randint(-100, 100))
        # Las fotos podrían tener distintas saturaciones  (valores entre -127 y 127. Fuente: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
        contraste = cv2.convertScaleAbs(image, alpha=random.uniform(1.0, 3.0))





def main():

    opt = get_args_parser()
    opt = opt.parse_args()
    csv_path = opt.csv_path
    output_path = opt.output_path
    data_path = opt.data_path
    # El "int(x)/100" lo hago para trabajar en tanto por uno en vez de porcentaje (es decir, por ejemplo, con 0,8 en vez de 80 %), se que es obvio, pero así me ahorro un momento de pensar.
    per_train, per_test, per_val = map(lambda x: int(x)/100, opt.train_test_val.split(','))
    dict_porcentajes_tipo = {TEST_FOLDER_PATH: per_train, TRAIN_FOLDER_PATH: per_test, VAL_FOLDER_PATH: per_val}
        
    # 1º Leer el CSV y obtener un diccionario de todos los elementos con el siguiente formato: "[tipo]: nombre".
    datos = leer_csv(csv_path, data_path)

    # 2º Realizar un aumento de datos (si es necesario)
    if opt.augment_data:
        data_augmentation(datos)
    
    # 3º Seleccionar un conjunto al azar en función de los parámetros:
    #   Porcentajes de Entrenamiento (80 %), Test (10 %) y Validación (10 %)
    datos_repartidos = repartir_datos(datos, dict_porcentajes_tipo)

    # 4º Guardar el listado de imagenes utilizadas en un fichero. ==> Mejor: que vaya al listado y vaya abriéndolas sin tener que separarlas (Mirar si hay algún problema con las librerías)   
    copiar_y_generar_imagenes(output_path, datos_repartidos)


if __name__ == '__main__':
    main()