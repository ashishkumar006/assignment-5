import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactMNIST(nn.Module):
    def __init__(self):
        super(CompactMNIST, self).__init__()
        # First convolution block
        self.conv1 = nn.Conv2d(1, 12, 3, stride=1, padding=1)  # 28x28x12
        self.bn1 = nn.BatchNorm2d(12)
        
        # Second convolution block with residual connection
        self.conv2a = nn.Conv2d(12, 16, 3, stride=2, padding=1)  # 14x14x16
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn2b = nn.BatchNorm2d(16)
        
        # Third convolution block
        self.conv3 = nn.Conv2d(16, 24, 3, stride=2, padding=1)  # 7x7x24
        self.bn3 = nn.BatchNorm2d(24)
        
        self.dropout = nn.Dropout2d(0.2)
        self.fc = nn.Linear(24 * 7 * 7, 10)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second block with residual
        identity = self.conv2a(x)  # Projection shortcut
        x = F.relu(self.bn2a(identity))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)  # Residual connection
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(-1, 24 * 7 * 7)
        x = self.fc(x)
        return x 