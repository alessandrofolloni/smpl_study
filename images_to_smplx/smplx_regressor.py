import torch
import torch.nn as nn


class SMPLXRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(SMPLXRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x