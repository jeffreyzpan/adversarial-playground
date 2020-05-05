import torch
import torch.nn as nn

class mnistCNN(nn.Module):

    def __init__(self, thermometer_encode=False, level=-1):
        super(mnistCNN, self).__init__()
        self.thermometer_encode = thermometer_encode
        if thermometer_encode:
            self.conv1 = nn.Conv2d(1*level, 32, kernel_size=5, stride=1, padding=1) 
        else:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.AdaptiveMaxPool2d((5,5))
        self.fc1 = nn.Linear(5* 5 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)
    
    def forward(self, x):
        #if self.thermometer_encode:
        #    x = torch.cat((x[0]), dim=1)

        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mnist_cnn(num_classes=10, thermometer_encode=False, level=-1):
    return mnistCNN(thermometer_encode, level)
