#!/bin/bash

#SBATCH --job-name=FT-50-época
#SBATCH --partition=P2
#SBATCH --nodelist=c03

#echo "Check cuda"

#srun nvidia-smi

echo "Fine_tuning"

echo -e "\n--"

cd ..

echo "Ejecución programa"

ruta_datasets="../Datasets"

#num_epochs=2
num_epochs=50
#layers_unfreeze=-1
name_dataset="Eyepacs_Aptos_Messidor"
#name_dataset="MESSIDOR2"
batch_size=64	#18
num_graficas=2
model="densenet169"
echo "num_epochs: $num_epochs"
#echo "layers_unfreeze: $layers_unfreeze"
echo "name_dataset: $name_dataset"
echo "batch_size: $batch_size"
echo "num_graficas: $num_graficas"
echo "model: $model"

#srun time python3 test.py  

# --layers_unfreeze = -1 ==> funciona normal, si no, se congelan el número de capas desde el final menos las indicadas.
#srun time python3 ./main.py \

# Por aquí torchrun no me va (desde el terminal directamente, sí)
srun time python3 -m torch.distributed.launch --use_env --nproc_per_node=$num_graficas ./main.py \
                            --batch_size $batch_size \
                            --epochs $num_epochs \
                            --device cuda  \
                            --data_path "$ruta_datasets/$name_dataset" \
                            --model $model \
                            --is_distributed    \
                            --patience 5
                            #--debug                            
                            #--model_weights "./model_vgg19_5_outputs.pth" \
