#!/bin/bash

#SBATCH --job-name=FT-2-épocas
#SBATCH --partition=P2
#SBATCH --nodelist=c03

echo "Fine_tuning"

echo -e "\n--"

cd ..

echo "Ejecución programa"

ruta_datasets="../Datasets"

num_epochs=2
#layers_unfreeze=-1
#name_dataset="Eyepacs_Aptos_Messidor"	#"MESSIDOR2"
name_dataset="MESSIDOR2"
batch_size=64	#18
echo "num_epochs: $num_epochs"
#echo "layers_unfreeze: $layers_unfreeze"
echo "name_dataset: $name_dataset"
echo "batch_size: $batch_size"

#srun time python3 test.py  

# --layers_unfreeze = -1 ==> funciona normal, si no, se congelan el número de capas desde el final menos las indicadas.
#srun time python3 ./main.py \
srun time python3 -m torch.distributed.launch --use_env --nproc_per_node=1 --master_port=48798 ./main.py \
                            --batch_size $batch_size \
                            --epochs $num_epochs \
                            --device cuda  \
                            --data_path "$ruta_datasets/$name_dataset" \
                            --model "resnet50" \
                            --is_distributed                            
                            #--model_weights "./model_vgg19_5_outputs.pth" \
