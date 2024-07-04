# TFM

## Instalación

Si tienes una versión de CUDA superior a 11.0:
```
conda create -n proyecto python=3.8 -y
conda activate proyecto
pip install -r requirements.txt
```
### Aún no va
Si tienes una versión de CUDA inferior o igual a 11.0 (es por si se utiliza en P2-c01):
```
conda create -n proyecto_CUDA11 python=3.8 -y
conda activate proyecto_CUDA11
pip install -r requirements_proyecto_CUDA11.txt
```

## Ejemplos de uso:
```
python3 ./main.py --batch_size 16 --epochs 1 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "resnet152" --num_classes 5 --model_weights "./partial_models/model_0_20240703_101219.pth" --eval
```

```
python3 ./main.py --batch_size 16 --epochs 50 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "resnet50" --num_classes 5 --model_weights "./model_vgg19_5_outputs.pth" 
```

```
python3 ./main.py --batch_size 16 --epochs 1 --device cpu --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2"
--model_weights "./model_vgg19_5_outputs.pth" --model "vgg19"
```


# Desinstalación
```
conda remove -n proyecto --all
```
Borrar todos los ficheros de esta carpeta.
