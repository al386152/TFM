from torchvision.models import vgg19, resnet50, resnet152, densenet121, densenet169
from logging import _levelToName, INFO

# Nombres argumentos del programa
BATCH_SIZE = "batch_size"
EPOCS = "epochs"
INPUT_SIZE = "input_size"
LEARNING_RATE = "lr"
DATA_PATH = "data_path"
NUMBER_CLASSES = "num_classes"
OUTPUT_DIR = "output_dir"
DEVICE = "device"
MODEL = "model"
PARTIAL_MODELS_PATH = "partial_models_path"
MODEL_WEIGHTS = "model_weights"
SHOW_DEBUG_OUTPUTS = "debug"
NOT_FREEZE_LAYERS = "not_freeze"
IS_DISTRIBUTED = "is_distributed"
EARLY_STOPPING_PATIENCE = "patience"
EARLY_STOPPING_MIN_DELTA = "min_delta"
COLOR_JITTER_BRIGHTNESS = "brightness"
COLOR_JITTER_CONTRAST = "contrast" 
COLOR_JITTER_SATURATION = "saturation"
COLOR_JITTER_HUE = "hue"
IS_HYPERTUNING = "hypertuning"
OPTIMIZER = "optimizer"

TEST_DATA_PATH = "test_path"
TRAIN_DATA_PATH = "train_path"
VALIDATION_DATA_PATH = "val_path"
SPLIT_PERCENTAGES = "split_perc"

TEST_FOLDER_NAME = "test"
VALIDATION_FOLDER_NAME = "val"
TRAIN_FOLDER_NAME = "train"
LIST_FOLDER_NAMES = [TEST_FOLDER_NAME, VALIDATION_FOLDER_NAME, TRAIN_FOLDER_NAME]

DEFAULT_TRAIN_VAL_TEST_PERCENTAGES=[0.7, 0.2, 0.1]

DEFAULT_LOG_FOLDER = "logs"
BASE_LOG_FILE_NAME = "output"

LOG_OUTPUT_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
LOG_BATCH_TRAINING_FORMAT = '\r' + LOG_OUTPUT_FORMAT

SEPARADOR_INPUT_IMAGENES = 'x'
SEPARADOR_INPUTS_COLOR_JITTER = ','
SEPARADOR_SPLIT_PERCENTAGES = ','

ANCHURA_IMG = "anchura"
ALTURA_IMG = "altura"

IMAGES_FOLDER_NAME = "result_images"
IMAGES_FILE_FORMAT = ".png"

NUM_GUIONES = 4

TANTO_POR_UNO_LOGS_PRINT = 0.1

# No me apetecía poner if-else para acabar haciendo lo mismo (además de que saco el listado de posibles modelos más facilmente)
SWITCH_MODELOS = {
    "vgg19": vgg19,
    "resnet50": resnet50,
    "resnet152": resnet152,
    "densenet121": densenet121,
    "densenet169": densenet169
}

POSSIBLE_MODELS = SWITCH_MODELOS.keys()
DEFAULT_MODEL_WEIGHTS = "IMAGENET1K_V1"

POSSIBLE_OPTIMIZERS = ["Adam", "AdamW", "SGD"]

BACKEND = "nccl" # de torch.distributed.init_process_group: ``mpi``, ``gloo``, ``nccl``, and ``ucc``

ROTATION_DEGREES = 75 # Diría que 30º en ambos sentidos es la rotación máxima que podría encontrarse en la vida real
RANDOM_PERSPECTIVE_DISTORSION_SCALE = 0.75 # No creo que sea buena idea modificar demasiado las imágenes
P_H_FLIP = 0.2
P_V_FLIP = 0.6


# Nombres de las Métricas:
AVG_LOSS = "Avg.Loss"
EPOCH_LOSS = "Epoch loss"

ACCURACY = "Accuracy"
AUROC = "AUROC"
AVERAGE_PRECISION = "Average Precision"
F_ONE_SCORE = "F1 Score"
CONFUSION_MATRIX = "Confusion Matrix"

MAIN_METRIC = ACCURACY

OPTUNA_SEED = 1234
OPTUNA_NUMBER_TRIALS = 30

# Nota: el timeout tiene que ser el mismo para todos los hilos ==> uno acaba, el resto espera al que ha acabado ==> no se acaba nunca.
#   Por otra parte, puesto que no sé bien cómo ponerlo, prefiero no ponerlo.
#OPTUNA_TIMEOUT = None #60 * 5 # Es en segundos (None: todo el tiempo que necesite)

# Esto no es del todo una constante, pero bueno:
loggin_level = _levelToName[INFO]
