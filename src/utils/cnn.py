import torch
import torch.nn as nn

class Text_CNN(nn.Module):
    def __init__(self, d_model, num_classes):
        super(Text_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(d_model, 256, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=2, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=4, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, padding=8, dilation=8)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.bn4 = nn.BatchNorm1d(1024)
        self.relu4 = nn.ReLU()
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.relu1(self.bn1(self.pool1(self.conv1(x))))
        x = self.relu2(self.bn2(self.pool2(self.conv2(x))))
        x = self.relu3(self.bn3(self.pool3(self.conv3(x))))
        x = self.relu4(self.bn4(self.pool4(self.conv4(x))))
        
        x = self.pool(x).squeeze(dim=2)
        
        x = self.dropout(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
