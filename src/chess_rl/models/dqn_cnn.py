import torch
import torch.nn as nn
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.utils.action_encoder import ACTION_SIZE

class DQNCNN(nn.Module):

    def __init__(self, action_size: int = ACTION_SIZE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=BOARD_CHANNELS,
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