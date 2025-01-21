#!/bin/bash

#SBATCH --job-name=Test-Inferencia-Ensemble
#SBATCH --partition=P2
#SBATCH --nodelist=c03

#S BATCH --gpus-per-task=1
#S BATCH --cpus-per-task=12
#S BATCH --mem=500G


echo "Inferencia - Ensemble"

echo -e "\n--"

cd ..

echo "Ejecución programa"

ruta_datasets="../Datasets"

num_epochs=1
#layers_unfreeze=-1
name_dataset="IDRiD"
#name_dataset="Eyepacs_Aptos_Messidor"
batch_size=48 #64
num_graficas=1
models="densenet169,densenet169,resnet50"
type_models="regression,regression,regression"
rutas_pesos="./pesos/regresion/regression_model_densenet169_MAE_0.8,1.8,2.3,3.3,4.pth@./pesos/regresion/regression_model_densenet169_MAE_0,1,2,3,4.pth@./pesos/regresion/regression_model_resnet50_MSE_0.3,1.3,2.3,3.3,4.pth"
optimizer="SGD,SGD,Adam"
votation_weights="0.96,0.66,0.4,0.57,0.61#0.94,0.62,0.57,0.5,0.54#0.61,0.59,0.46,0.64,0.54"
class_boundries="0.8,1.8,2.3,3.3,4#0,1,2,3,4#0.3,1.3,2.3,3.3,4"

echo "num_epochs: $num_epochs"
#echo "layers_unfreeze: $layers_unfreeze"
echo "name_dataset: $name_dataset"
echo "batch_size: $batch_size"
echo "num_graficas: $num_graficas"
echo "models: $models"
echo "type_models: $type_models"
echo "rutas_pesos: $rutas_pesos"
echo "votation_weights: $votation_weights"
echo "optimizer: $optimizer"
echo "class_boundries: $class_boundries"

#srun time python3 test.py  

# --layers_unfreeze = -1 ==> funciona normal, si no, se congelan el número de capas desde el final menos las indicadas.
#srun time python3 ./main.py \

# Por aquí torchrun no me va (desde el terminal directamente, sí)
#srun time python3 -m torch.distributed.launch --use_env --nproc_per_node=$num_graficas ./main.py \
time python3 -m torch.distributed.launch --use_env --nproc_per_node=$num_graficas ./main.py \
    --batch_size $batch_size \
    --epochs $num_epochs \
    --device cuda  \
    --data_path "$ruta_datasets/$name_dataset" \
    --model $models \
    --num_classes 5 \
    --optimizer $optimizer \
    --input_size "524" \
    --votation_weights $votation_weights \
    --model_weights "$rutas_pesos" \
    --inference \
    --is_distributed \
    --type_models $type_models \
    --class_boundries $class_boundries \
#--loss_function CrossEntropyLoss \

