import torch
import torch.nn as nn
import torchvision.models as models


# 1 mlp per car, multi label prediction
class MultiLabelNeuralNetwork(nn.Module):
    def __init__(self, backbone, dim_out):
        super(MultiLabelNeuralNetwork, self).__init__()
        # resnet = models.resnet18(pretrained=True)
        resnet = backbone
        layers = list(resnet.children())[:9]
        self.fc = resnet.fc
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.classifier = nn.ModuleList()
        # shape output, 5 different shape + absence of car
        # length output, 2 different shape + absence of car
        # wall output, 2 different walls + absence of car represented as index 0
        # roof output, 4 different roof shapes + absence of car
        # wheels output, 2 different wheel counts + absence of car
        # load number output, max 3 payloads min 0
        # load shape output, 6 different shape + absence of car
        if dim_out == 28:
            self.label_num_classes = [6, 3, 3, 5, 3, 4, 7]
        else:
            # all labels can obtain all classes
            self.label_num_classes = [22] * 8
        all_classes = sum(self.label_num_classes)
        in_features = resnet.inplanes
        for _ in range(4):
            self.classifier.append(nn.Sequential(nn.Linear(in_features=in_features, out_features=all_classes)))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)

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
