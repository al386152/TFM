import sys

import torch
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import vgg19, resnet50, resnet152, densenet121, densenet169

SWITCH_MODELOS = {
    "vgg19": vgg19,
    "resnet50": resnet50,
    "resnet152": resnet152,
    "densenet121": densenet121,
    "densenet169": densenet169
}

def main(argv):
    model = DistributedDataParallel(SWITCH_MODELOS[argv[1]])
    model.load_state_dict(torch.load(argv[1], map_location=torch.device('cpu')))
    torch.save(model.module.state_dict(), argv[2])


# argv[1] modelo
# argv[2] path pesos DistributedDataParallel
# argv[3] path pesos Bien hechos.

if __name__ == "__main__":
    argv = sys.argv
    print(f"argv: {argv}")
    main(argv)