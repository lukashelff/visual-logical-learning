import torch.nn as nn
import torchvision.models as models

from util import *

PYTORCH_VER = torch.__version__


# NS-VQA code for spacial attribute detection adopted by lukas helff
# https://github.com/kexinyi/ns-vqa/blob/master/scene_parse/attr_net/model.py


class PositionNetwork(nn.Module):
    def __init__(self, dim_input=4, dim_output=3):
        super(PositionNetwork, self).__init__()

        resnet = models.resnet50(pretrained=True)
        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 4-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(dim_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)

        in_features = resnet.inplanes
        self.classifier = nn.Sequential(nn.Linear(in_features=in_features, out_features=dim_output))

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        # x = self.final_layer(x)
        # soft = nn.Softmax(dim=1)
        class_output = self.classifier(x)
        return class_output
