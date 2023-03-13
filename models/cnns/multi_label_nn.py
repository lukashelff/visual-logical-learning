import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms


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
        elif dim_out == 32:
            # all labels can obtain all classes
            self.label_num_classes = [22] * 8
        else:
            raise ValueError(f'unknown dim_out {dim_out}')
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
        # self.predict_train(preds)

        return preds

    def predict_train(self, activations):

        preds = torch.max(activations, dim=-1)[1].detach().cpu().numpy()
        print_train(preds)


def print_train(outputs):
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
    attributes = ['color', 'length', 'wall', 'roof', 'wheels', 'load1', 'load2', 'load3']
    preds = torch.max(outputs, dim=2)[1]
    preds = preds.T if preds.shape[0] == 32 else preds
    for i in range(preds.shape[0]):
        print("Train", i)
        for j in range(preds.shape[1] // 8):
            car = 'Car' + str(j) + ': '
            for k in range(8):
                car += attributes[k] + '(' + attribute_classes[preds[i, j * 8 + k]] + f'{preds[i, j * 8 + k]})'
                car += ', ' if k < 7 else ''
            print(car)


def show_torch_im(x):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    from matplotlib import pyplot as plt
    plt.imshow(invTrans(x)[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.savefig("im1.png")
    plt.show()