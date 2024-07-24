# TFM

## Instalación

```
conda create -n proyecto python=3.8 -y
conda activate proyecto
conda install cudatoolkit=12 -c pytorch -y
pip install -r requirements.txt
```

## Ejemplos de uso:
```
python3 ./main.py --batch_size 64 --epochs 1 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "vgg19" --num_classes 5 --debug
```

```
python3 ./main.py --batch_size 16 --epochs 50 --device cpu  --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/MESSIDOR2" --model "resnet50" --num_classes 5 --model_weights "./model_vgg19_5_outputs.pth" 
```

```
python3 ./main.py --batch_size 64 --epochs 1 --device cpu --data_path "/home/usuario/Documentos/Retinopatía Diabética/Datasets/IDRiD/train"  --model "resnet50" --num_classes 5 --split_perc "60,20,20" --hypertuning
```

Para realizar el entrenamiento entre varias GPUS
```
python3 -m torch.distributed.launch --use_env --nproc_per_node=$num_graficas ./main.py \
                            --batch_size 64 \
                            --epochs 50 \
                            --device cuda  \
                            --data_path "../Datasets/Eyepacs_Aptos_Messidor" \
                            --model "densenet169" \
                            --is_distributed \
                            --patience 5
```

# Desinstalación
```
conda remove -n proyecto --all
```
Borrar todos los ficheros de esta carpeta.
