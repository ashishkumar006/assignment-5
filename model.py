import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactMNIST(nn.Module):
    def __init__(self):
        super(CompactMNIST, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        
        # Residual block
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Projection shortcut
        self.shortcut = nn.Conv2d(12, 16, kernel_size=1)
        
        # Final layers
        self.conv4 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(24 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        identity = self.shortcut(x)
        
        # Residual block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity)  # Residual connection
        
        # Final convolution
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 4)
        
        # Fully connected
        x = x.view(-1, 24 * 7 * 7)
        x = self.dropout(x)
        x = self.fc(x)
        return x