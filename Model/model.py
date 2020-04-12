import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
    # use this model for checkpoints following the format: chcekpointx.pth where x [1,5]
    # this model has given good results 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(5*5*64, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""

class Net(nn.Module):
    # use this with checkpoint_new1.pth
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9*9*64, 500)
        self.dropout1 = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        # flattening the image
        x = x.view(-1, 9*9*64)
        x = self.dropout1(F.relu(self.fc1(x)))
        # final output
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# use this for checkpoint_new2.pth
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5, stride=1)
#         self.norm1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 48, 5, 1)
#         self.norm2 = nn.BatchNorm2d(48)
#         self.fc1 = nn.Linear(9*9*48, 420)
#         self.dropout1 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(420, 10)

#     def forward(self, x):
#         x = F.relu(self.norm1(self.conv1(x)))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.norm2(self.conv2(x)))
#         x = F.max_pool2d(x, 2, 2)
#         # flattening the image
#         x = x.view(-1, 9*9*48)
#         x = self.dropout1(F.relu(self.fc1(x)))
#         # final output
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

def load_model(path= "./Weights/checkpoint_new2.pth"):
    print("constructing the neural network.....")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    checkpoint = torch.load(path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # set model to eval mode, this takes care of dropout stuff
    model.eval()
    return model