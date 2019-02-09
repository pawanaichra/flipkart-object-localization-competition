import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=3)
        self.conv2 = nn.Conv2d(6, 8, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=3)   
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=3) 
        self.fc1 = nn.Linear(5040, 1000*3)
        self.fc2 = nn.Linear(1000*3, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)      
        x = x.view(-1, 5040)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        m = torch.tensor([[640, 480, 640, 480]]).type(torch.DoubleTensor).cuda()
        x = x*m
        return x
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))