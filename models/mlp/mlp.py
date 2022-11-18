import torch
import torch.nn as nn
import torch.nn.functional as F  # define NN architecture


class MLP(nn.Module):
    def __init__(self, dim_in=4 * 32, dim_out=22 * 32):
        super().__init__()

        # linear layer dim_in = 4 (class label,x,y,z) * 32 (label count)
        self.fc1 = nn.Linear(dim_in, 100)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(100, 200)
        # linear layer dim_out = 22 (classes per label) * 32 (label count)
        self.fc3 = nn.Linear(200, dim_out)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)
        self.label_num_classes = [22] * 8 * 4
        self.all_classes = sum(self.label_num_classes)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        # # flatten image input
        x = x.view(-1, 4 * 32)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        x = x.view(-1, 32, 22)
        # x = self.soft(x)
        return x
