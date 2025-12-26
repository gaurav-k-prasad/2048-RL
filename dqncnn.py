import torch.nn as nn
import torch.nn.functional as F


class QNetworkCNN(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256), nn.ReLU(), nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # x: (B, 1, 4, 4)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
