# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models


class Resnet(nn.Module):
    def __init__(self,version=18):
        super(Resnet, self).__init__()
        if version == 18:
            self.feature = models.resnet18(pretrained=True)
        elif version == 34:
            self.feature = models.resnet34(pretrained=True)
        if version == 50:
            self.feature = models.resnet50(pretrained=True)
        if version == 101:
            self.feature = models.resnet101(pretrained=True)
            
        # # Freeze the weights of the initial convolutional layers (e.g., the first two layers)
        # for param in self.feature.parameters():
        #     param.requires_grad = False
    
        num_ftrs = self.feature.fc.in_features
        self.feature.fc = nn.Linear(num_ftrs, 1)
                
    def forward(self, input_data):
        return self.feature(input_data)


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.feature = models.mobilenet_v2(pretrained=True)
            
        # # Freeze the weights of the initial convolutional layers (e.g., the first two layers)
        # for param in self.feature.parameters():
        #     param.requires_grad = False
    
        num_ftrs = self.feature.classifier[1].in_features
        self.feature.classifier[1] = nn.Linear(num_ftrs, 1)
                
    def forward(self, input_data):
        return self.feature(input_data)