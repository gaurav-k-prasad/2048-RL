import torch
import torch.nn as nn

class DQN(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.fc1 = nn.Linear(16, 8)
    self.fc2 = nn.Linear(8, 4)

  def forward(self, x) -> torch.Tensor:
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  