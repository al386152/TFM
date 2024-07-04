from torchvision.models import vgg19, resnet50, resnet152
from logging import _levelToName, INFO

# Nombres argumentos
NAME_BATCH_SIZE = "batch_size"
NAME_EPOCS = "epochs"
NAME_INPUT_SIZE = "input_size"
NAME_DROP_PATH = "drop_path"
NAME_WEIGTH_DECAY = "weight_decay"
NAME_LEARNING_RATE = "lr"
NAME_BASE_LEARNING_RATE = "blr"
NAME_WARMUP_EPOCHS = "warmup_epochs"
NAME_DATA_PATH = "data_path"
NAME_NUMBER_CLASSES = "num_classes"
NAME_OUTPUT_DIR = "output_dir"
NAME_DEVICE = "device"
NAME_MODEL = "model"
PARTIAL_MODELS_PATH = "partial_models_path"
NAME_MODEL_WEIGHTS = "model_weights"
NAME_SHOW_DEBUG_OUTPUTS = "debug"

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

SWITCH_MODELOS = {
    "vgg19": vgg19,
    "resnet50": resnet50,
    "resnet152": resnet152
}

POSSIBLE_MODELS = SWITCH_MODELOS.keys()
DEFAULT_MODEL_WEIGHTS = "IMAGENET1K_V1"

# Esto no es del todo una constante, pero bueno:
loggin_level = _levelToName[INFO]

    
