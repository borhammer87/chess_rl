import torch

from chess_rl.models.dqn_cnn import DQNCNN


def test_dqn_cnn_output_shape():

    model = DQNCNN()

    x = torch.randn(1, 12, 8, 8)

    output = model(x)

    assert output.shape == (1, 4096)