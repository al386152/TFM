# utils
Realmente, aquí es en donde están programadas todas las funcionalidades del código.

Nota: mirar de cambiar el nombre a la carpeta

## arguments_parser.py
En este fichero se encuentran las funciones para convertir los parámetros en elementos de un diccionario, además de comprobar que sean correctos y adaparlos de la sintaxis utilizada en el terminal a Python (solo unos pocos parámetros lo necesitan).

Depende de "constants.py" para los nombres de los parámetros y las claves del diccionario de parámetros, y de "log_writer.py" para mostrar la información.

## constants.py
En este fichero se declaran todas las constantes del programa, por ejemplo, todos los nombres del diccionario, nombres de los ficheros de salida o la semilla de aleatoriedad.

## data_loaders.py
En este fichero se recoge todo el código relacionado con la carga de los datos, sus transformaciones y la generación de los "samplers".

## ensemble.py
En este fichero se recoge todo el código relacionado con la generación de "Ensembles", es decir, de modelos formados por un conjunto de otros modelos.

## log_writer.py
En este fichero se encuentra el código relacionado con los "logs", incluyendo la redirección de las salidas a un fichero y a la salida estándar o el establecimiento del nivel del "log" (información, debug, etc.).

## metrics.py
En este fichero se declaran las funciones relacionadas con las mediciones de los resultados.

## model_related.py
En este fichero se recoge todo aquello relacionado con el modelo (como la creación del objeto de modelo o la evaluación del modelo) menos el entrenamiento, que está en "training.py" (se hacía demasiado largo el fichero).

## modified_image_folder.py
En este fichero se recoge una modificación de la clase "ImageFolder" para poder aplicar una lista de transformacióones u otra en función de si pertenecen a una clase o a otra.

## operations.py
En este fichero se encuentran distintas funciones que, o bien se utilizan en varios ficheros, o no cuadraban en ninún otro fichero.

## optuna_related.py
En este fichero se está todo el código para la búsqueda de los mejores hiperparámetros haciendo uso de las bibliotecas de "Optuna".

## training.py
En este fichero se encuentra todo el código relacionado con el entrenamiento.

