
import torch
from torch import nn


class CaptureFeatureQ(nn.Module):
    """Diagnostic Q-network using captured-piece features."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 6:
            raise ValueError("Expected features with shape (batch, 6).")

        return self.network(features.float()).squeeze(-1)
