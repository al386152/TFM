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

```
srun time python3 -m torch.distributed.launch --nproc_per_node=$num_graficas --use_env ./main.py \
                            --batch_size $batch_size \
                            --epochs $num_epochs \
                            --device cuda  \
                            --data_path "$ruta_datasets/$name_dataset" \
                            --model $model \
                            --hypertuning  \
                            --is_distributed
```

Para ejecutar un "ensemble":
```
time python3 -m torch.distributed.launch --nproc_per_node=1 --use_env ./main.py \
                            --batch_size 48 \
                            --epochs 2 \
                            --device cpu  \
                            --data_path "../Datasets/Eyepacs_Aptos_Messidor \
                            --model "densenet169,resnet50" \
                            --num_classes 5 \
                            --loss_function CrossEntropyLoss \
                            --patience 3 \
                            --optimizer "SGD" \
                            --is_distributed \
                            --lr 0.00899688614556782 \
                            --hue 0 \
                            --saturation 0.1 \
                            --contrast 1 \
                            --brightness 0.4 \
                            --rotation 0 \
                            --random_distorsion 0.3 \
                            --horizontal_flip 0.2 \
                            --vertical_flip 0.4 \
                            --input_size "524" \
                            --batch_augmentation 1 \
                            --p_dropout 0
```

```
time python3 ./main.py \
    --batch_size 48 \
    --epochs 2 \
    --device cpu  \
    --data_path "../Datasets/Eyepacs_Aptos_Messidor" \
    --model "densenet169,resnet50" \
    --num_classes 5 \
    --loss_function CrossEntropyLoss \
    --patience 3 \
    --optimizer "SGD" \
    --lr 0.00899688614556782 \
    --hue 0 \
    --saturation 0.1 \
    --contrast 1 \
    --brightness 0.4 \
    --rotation 0 \
    --random_distorsion 0.3 \
    --horizontal_flip 0.2 \
    --vertical_flip 0.4 \
    --input_size "524" \
    --batch_augmentation 1 \
    --p_dropout 0 \
    --votation_weights 1,0.8,0.7,0.5,0.6#1.2,0.4,0.9,0.1,0.2
```

# Desinstalación
```
conda remove -n proyecto --all
```
Borrar todos los ficheros de esta carpeta.
