import torch.nn as nn

class Emo2CNN(nn.Module):
    def __init__(self):
        super(Emo2CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0), # (16, 46, 46)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # (32, 44, 44)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (32, 22, 22)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), # (64, 20, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (64, 10, 10)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 10, 10)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 10, 1)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 10 * 10, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 46, 46)
        x = self.layer2(x) # (32, 22, 22)
        x = self.layer3(x) # (64, 10, 10)
        x = self.layer4(x) # (128, 10, 10)
        x = self.layer5(x) # (128, 10, 10)
        x = x.reshape(x.size(0), -1) # (128 * 10 * 10)
        x = self.fc(x)
        return x
