import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvNetv2(nn.Module):
    def __init__(self, n_classes):
        super(EnvNetv2, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, (1, 64), stride=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 16), stride=(1, 2)), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(1, 32, (8, 8)), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 32, (8, 8)), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(32, 64, (1, 4)), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, (1, 4)), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 128, (1, 2)), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, (1, 2)), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(128, 256, (1, 2)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 256, (1, 2)), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fc11 = nn.Linear(256 * 10 * 8, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, n_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, (1, 64))
        x = x.permute(0, 2, 1, 3)

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, (5, 3))
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, (1, 2))
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(x, (1, 2))
        x = self.conv9(x)
        x = self.conv10(x)
        x = F.max_pool2d(x, (1, 2))
        x = torch.flatten(x, start_dim=1)

        x = F.dropout(F.relu(self.fc11(x)))
        x = F.dropout(F.relu(self.fc12(x)))

        return self.fc13(x)

def envnetv2(num_classes=50):
    return EnvNetv2(num_classes)
