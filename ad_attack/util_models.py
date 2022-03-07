# install dependencies
import torch
import torchvision
from fastai.vision.all import *
from fastai.vision import *
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np
from n2gem.metrics import gem_build_coverage, gem_build_density
import eagerpy as ep
import matplotlib.pyplot as plt

class CNet(nn.Module):
    """
    The model architecture for the over-fit model
    
    # for overfit model: dropout is removed and trained for 50 epochs
    The results are not included in this study, hence they can be avoided
    """
    
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7)
        self.fc1 = nn.Linear(4*11*11, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = x.view(-1, 4*11*11)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

   
class UFNet(nn.Module):
    """
    The model architecture for the under-fit model
    
    The results are not included in this study, hence they can be avoided
    """
    def __init__(self):
        super(New, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(32*13*13, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 14)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        return x
   
class FXMNIST(Mnet):
    """
    The model architecture for the feature extraction of MNIST dataset
    
    The results are not included in this study, hence they can be avoided
    """
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x1 = self.drop(F.max_pool2d(x, 2))
        x2 = F.relu(self.conv2(x1))
        x3 = F.max_pool2d(x2, 2)
        x4 = x3.view(-1, 800)
        x5 = self.drop(F.relu(self.fc1(x4)))
        x6 = F.relu(self.fc2(x5))
        x7 = self.fc3(x6)
        return x7
    
class FXMed(nn.Module):
    """
    The model architecture for the feature extraction of MedMNIST dataset
    
    The results are not included in this study, hence they can be avoided
    """
    def __init__(self, in_features, op_features):
        super(FXMed, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), # b4- conv5 output 64*4*4
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, op_features))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x