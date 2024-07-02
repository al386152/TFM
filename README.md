# TFM

## Instalación
```
conda create -n proyecto python=3.8 -y
conda activate proyecto
pip install -r requirements.txt
```

## Ejemplos de uso:
```
python3 ./main.py --batch_size 16 --epochs 1 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "resnet152" --num_classes 5
```

```
python3 ./main.py --batch_size 16 --epochs 50 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "resnet50" --num_classes 5 --model_weights "./model_vgg19_5_outputs.pth" 
```

````
python3 ./main.py --batch_size 16 --epochs 1 --device cpu --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2"
--model_weights "./model_vgg19_5_outputs.pth" --model "vgg19"
```