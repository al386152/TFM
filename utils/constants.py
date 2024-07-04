from torchvision.models import vgg19, resnet50, resnet152, densenet121, densenet169
from logging import _levelToName, INFO

# Nombres argumentos
BATCH_SIZE = "batch_size"
EPOCS = "epochs"
# TODO: Utilizar los siguientes parámetros :)
INPUT_SIZE = "input_size"   
DROP_PATH = "drop_path"
WEIGTH_DECAY = "weight_decay"
LEARNING_RATE = "lr"
BASE_LEARNING_RATE = "blr"
WARMUP_EPOCHS = "warmup_epochs"
NAME_DATA_PATH = "data_path"
NUMBER_CLASSES = "num_classes"
OUTPUT_DIR = "output_dir"
DEVICE = "device"
MODEL = "model"
PARTIAL_MODELS_PATH = "partial_models_path"
MODEL_WEIGHTS = "model_weights"
SHOW_DEBUG_OUTPUTS = "debug"
NOT_FREEZE_LAYERS = "not_freeze"

TEST_FOLDER_NAME = "test"
VALIDATION_FOLDER_NAME = "val"
TRAIN_FOLDER_NAME = "train"
LIST_FOLDER_NAMES = [TEST_FOLDER_NAME, VALIDATION_FOLDER_NAME, TRAIN_FOLDER_NAME]

DEFAULT_LOG_FOLDER = "logs"
BASE_LOG_FILE_NAME = "output"

LOG_OUTPUT_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
LOG_BATCH_TRAINING_FORMAT = '\r' + LOG_OUTPUT_FORMAT

WIDTH_IMAGES = 512
HEIGHT_IMAGES = 512

NUM_GUIONES = 4

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

# Esto no es del todo una constante, pero bueno:
loggin_level = _levelToName[INFO]

    
