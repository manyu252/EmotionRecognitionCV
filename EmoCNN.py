import torch.nn as nn

class EmoCNN(nn.Module):
    def __init__(self):
        super(EmoCNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (1, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 12 * 12, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = x.reshape(x.size(0), -1) # (128 * 12 * 12)
        x = self.fc(x)
        return x
