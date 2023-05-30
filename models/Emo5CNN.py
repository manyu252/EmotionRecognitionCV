import torch.nn as nn

class Emo5CNN(nn.Module):
    def __init__(self):
        super(Emo5CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (64, 12, 12)
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
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 12 * 12, 7) # fc 64 * 12 * 12 -> 7

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = self.layer6(x) # (256, 12, 12)
        x = self.layer7(x) # (256, 12, 12)
        x = self.layer8(x) # (128, 12, 12)
        x = self.layer9(x) # (64, 12, 12)
        x = x.reshape(x.size(0), -1) # (64 * 12 * 12)
        x = self.fc1(x)
        return x