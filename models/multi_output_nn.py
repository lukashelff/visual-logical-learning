import torch
import torch.nn as nn
import torchvision.models as models


# 1 mlp for every label
class MultiOutputNeuralNetwork(nn.Module):
    def __init__(self):
        super(MultiOutputNeuralNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        layers = list(resnet.children())[:9]
        self.fc = resnet.fc
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.classifier = nn.ModuleList()
        for _ in range(4):
            # shape output, 5 different shape + absence of car
            # length output, 2 different shape + absence of car
            # roof output, 4 different roof shapes + absence of car
            # wheels output, 2 different wheel counts + absence of car
            # load number output, max 3 payloads min 0
            # load shape output, 6 different shape + absence of car
            for num_feature in [6, 3, 5, 3, 4, 7]:
                self.classifier.append(
                    nn.Sequential(nn.Linear(in_features=512, out_features=num_feature),
                                  nn.Softmax(dim=1)))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        # x = F.relu(x)
        # x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # x = x.view(x.shape[0], -1)
        # outputs = (classifier(x) for classifier in self.classifier)
        # for classifier in self.classifier:
        #     out = classifier(x).unsqueeze(dim=0)
        #     outputs = torch.cat((outputs, out), dim=0)
        # outputs = self.classifier[0](x)
        return [classifier(x) for classifier in self.classifier]
