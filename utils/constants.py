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
SUMMARIES_PATH = "summaries"
NAME_MODEL_WEIGHTS = "model_weights"

TEST_FOLDER_NAME = "test"
VALIDATION_FOLDER_NAME = "val"
TRAIN_FOLDER_NAME = "train"
LIST_FOLDER_NAMES = [TEST_FOLDER_NAME, VALIDATION_FOLDER_NAME, TRAIN_FOLDER_NAME]

POSSIBLE_MODELS = ["vgg19", "resnet50"]
DEFAULT_MODEL_WEIGHTS = "IMAGENET1K_V1"

WIDTH_IMAGES = 512
HEIGHT_IMAGES = 512

NUM_GUIONES = 16

#Esto es lo más similar a una macro de C
def OUTPUT_MODEL_NAME(name, number_clases):
    return f"model_{name}_{number_clases}_outputs.pth"