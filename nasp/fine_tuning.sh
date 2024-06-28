#!/bin/bash

#SBATCH --job-name=FT-50-épocas
#SBATCH --partition=P2
#SBATCH --nodelist=c02

echo "Fine_tuning"

echo -e "\n--"
echo "Ejecución programa"

cd ..

ruta_datasets="../Datasets"

num_epochs=50
#layers_unfreeze=-1
#name_dataset="Eyepacs_Aptos_Messidor"	#"MESSIDOR2"
name_dataset="MESSIDOR2"
batch_size=64	#18
echo "num_epochs: $num_epochs"
#echo "layers_unfreeze: $layers_unfreeze"
echo "name_dataset: $name_dataset"
echo "batch_size: $batch_size"

# --layers_unfreeze = -1 ==> funciona normal, si no, se congelan el número de capas desde el final menos las indicadas.
srun time python3 ./main.py --batch_size $batch_size --epochs $num_epochs --device cuda  --data_path "$ruta_datasets/$name_dataset"
