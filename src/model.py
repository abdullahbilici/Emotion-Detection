import torch
from torch import nn
from torch.nn import functional as F



class CNN128(nn.Module):
    def __init__(self, number_of_emotions = 4):
        super(CNN128, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Linear(16 * 32 * 32, 64)
        self.dropout = nn.Dropout(p = 0.35)
        self.fc2 = nn.Linear(64, number_of_emotions)

    def forward(self, x):
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    

class CNN48(nn.Module):
    def __init__(self, number_of_emotions = 4):
        super(CNN48, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = .4),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = .4),


            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            

            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(p = .5),

        )

        self.dense_layers = nn.Sequential(
            nn.Linear(128 * 12 * 12, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p = 0.6),
            nn.Linear(128, number_of_emotions),

        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)

        x = self.dense_layers(x)
        return x