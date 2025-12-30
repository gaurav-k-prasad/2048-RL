import torch
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
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        board = x[:, :, :4, :]
        mask = x[:, :, -1, :].squeeze(1)
        
        mask.to(x.device)
        board.to(x.device)

        features = self.conv(board)
        features = features.view(features.size(0), -1)
        q_values = self.fc(features)
        masked_q_values = q_values + (mask - 1) * 1e9
        return masked_q_values
