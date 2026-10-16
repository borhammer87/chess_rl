
import pytest
import torch

from chess_rl.models.capture_feature_q import CaptureFeatureQ


def test_capture_feature_q_scores_batched_actions():
    model = CaptureFeatureQ()
    features = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    scores = model(features)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_capture_feature_q_supports_gradient_updates():
    model = CaptureFeatureQ()
    features = torch.eye(6)
    targets = torch.tensor([0.01, 0.03, 0.03, 0.05, 0.09, 0.00])

    predictions = model(features)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()

    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )


def test_capture_feature_q_rejects_invalid_shape():
    model = CaptureFeatureQ()

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(6))
