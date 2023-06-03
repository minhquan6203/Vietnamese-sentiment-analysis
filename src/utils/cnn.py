import torch
import torch.nn as nn

class Text_CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super( Text_CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128, num_classes)  # num_classes là số lượng lớp phân loại của bạn

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


