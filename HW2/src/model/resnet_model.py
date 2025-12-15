import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # ⚠️ Исправлено: stride=1
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)  
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_layer(64, 64, 1, stride=1)
        self.block2 = self._make_layer(64, 128, 1, stride=2)
        self.block3 = self._make_layer(128, 256, 1, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten(),
            nn.Dropout(0.3),             
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.classifier(out)
        return out

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None

        if in_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
       
        for _ in range(1, blocks):  
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, downsample=None))
        
        return nn.Sequential(*layers)  