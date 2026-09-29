import torch

from model.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from model.losses import router_loss
from model.router import AdaptiveRouter


def _coeffs():
    return {"lambda_action": 1.0, "lambda_tool": 1.0,
            "lambda_terminal": 1.0, "lambda_conf": 0.5, "lambda_budget": 0.0}


def test_router_loss_scalar_and_backward():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    out = router(torch.randn(6, 32), torch.rand(6, 4))
    labels = {
        "action": torch.tensor([0, 1, 2, 1, 0, 1]),
        "tool": torch.tensor([-1, 0, -1, 2, -1, -1]),
        "terminal": torch.tensor([0, 0, 1, 0, 2, 0]),
        "confidence": torch.rand(6),
    }
    parts = router_loss(out, labels, _coeffs())
    assert parts["loss"].requires_grad
    assert "action_loss" in parts and "tool_loss" in parts
    parts["loss"].backward()


def test_tool_loss_masked_when_not_tool():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=16, mlp_hidden=8, mlp_out=8)
    out = router(torch.randn(4, 16), torch.rand(4, 4))
    labels = {
        "action": torch.zeros(4, dtype=torch.long),
        "tool": torch.full((4,), -1, dtype=torch.long),
        "terminal": torch.zeros(4, dtype=torch.long),
        "confidence": torch.rand(4),
    }
    parts = router_loss(out, labels, _coeffs())
    assert float(parts["tool_loss"]) == 0.0


def test_ece_bounds():
    conf = torch.tensor([0.9, 0.9, 0.1, 0.1])
    correct = torch.tensor([1, 1, 0, 0])
    ece = expected_calibration_error(conf, correct, n_bins=2)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    labels = torch.tensor([0, 1])
    t = fit_temperature(logits, labels, lr=0.1, steps=50)
    assert t > 0
    scaled = apply_temperature(logits, t)
    assert scaled.shape == logits.shape
    assert torch.allclose(scaled, logits / t, atol=1e-5)


def test_tool_loss_zero_when_tool_unlabeled():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=16, mlp_hidden=8, mlp_out=8)
    out = router(torch.randn(1, 16), torch.rand(1, 4))
    labels = {
        "action": torch.tensor([1]),
        "tool": torch.tensor([-1]),
        "terminal": torch.tensor([0]),
        "confidence": torch.tensor([0.5]),
    }
    parts = router_loss(out, labels, _coeffs())
    assert float(parts["tool_loss"]) == 0.0


def test_fit_temperature_adversarial():
    logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]])
    correct = torch.tensor([1, 1])
    t = fit_temperature(logits, correct)
    assert float(t) > 0 and torch.isfinite(torch.as_tensor(t))
    t2 = fit_temperature(logits, torch.tensor([0, 0]))
    assert float(t2) > 0 and torch.isfinite(torch.as_tensor(t2))


def test_ece_exact_and_empty():
    assert expected_calibration_error(torch.tensor([]), torch.tensor([], dtype=torch.long)) == 0.0
    ece = expected_calibration_error(torch.tensor([0.6, 0.6]), torch.tensor([1, 0]), n_bins=10)
    assert abs(ece - 0.1) < 1e-6


def test_utility_weighted_action_loss():
    from model.losses import utility_weighted_action_loss

    logits = torch.zeros(2, 3, requires_grad=True)
    targets = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 1.0])
    loss = utility_weighted_action_loss(logits, targets, weights)
    assert torch.isfinite(loss) and loss.requires_grad
    assert abs(float(loss.detach()) - float(torch.log(torch.tensor(3.0)))) < 1e-6


def test_ece_shape_mismatch_raises():
    try:
        expected_calibration_error(torch.rand(3), torch.ones(2, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fit_temperature_normalizes_labels():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    labels = torch.tensor([0, 1], dtype=torch.int32)
    t = fit_temperature(logits, labels, lr=0.1, steps=50)
    assert float(t) > 0 and torch.isfinite(torch.as_tensor(t))
