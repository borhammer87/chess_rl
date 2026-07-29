import torch
import torch.nn as nn


class DQNCNN(nn.Module):

    def __init__(self, action_size: int = 4096):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),

            nn.Linear(512, action_size)
        )

    def forward(self, x):

        x = x.float()

        x = self.features(x)

        x = self.classifier(x)

        return x