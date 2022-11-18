import torch.nn as nn
import torchvision.models as models

from util import *

PYTORCH_VER = torch.__version__


# NS-VQA code for spacial attribute detection adopted by lukas helff
# https://github.com/kexinyi/ns-vqa/blob/master/scene_parse/attr_net/model.py


class AttributeNetwork(nn.Module):
    def __init__(self, dim_input=32):
        super(AttributeNetwork, self).__init__()

        resnet = models.resnet50(pretrained=True)
        # resnet = models.resnet18(pretrained=True)
        layers = list(resnet.children())

        # remove the last layer
        layers.pop()
        # remove the first layer as we take a 64-channel input
        layers.pop(0)
        layers.insert(0, nn.Conv2d(dim_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)
        # self.final_layer = nn.Linear(512, output_dim)
        self.label_num_classes = [22] * 8
        all_classes = sum(self.label_num_classes)
        self.classifier = nn.ModuleList()
        # self.classifier.append(nn.Sequential(nn.Linear(in_features=512, out_features=all_classes)))
        in_features = resnet.inplanes
        for _ in range(4):
            self.classifier.append(nn.Sequential(nn.Linear(in_features=in_features, out_features=all_classes)))

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        # x = self.final_layer(x)
        soft = nn.Softmax(dim=1)
        class_output = [classifier(x) for classifier in self.classifier]
        preds = torch.cat(class_output, dim=1).view(-1, 32, 22)
        # preds = []
        # for output in class_output:
        #     for i, num_classes in enumerate(self.label_num_classes):
        #         ind_start = sum(self.label_num_classes[:i])
        #         ind_end = ind_start + num_classes
        #         pred = soft(output[:, ind_start:ind_end])
        #         preds.append(pred)
        return preds
