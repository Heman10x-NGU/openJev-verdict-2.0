"""Unit tests for proper scoring rules, temperature scaling, and ECE."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core.calibration import (
    TemperatureCalibrator,
    brier_score_loss,
    composite_loss,
    compute_ece,
    cross_entropy_loss,
)


def test_brier_score_loss_exact_values() -> None:
    # 2 classes: logits yielding [0.8, 0.2]
    # ln(0.8/0.2) = ln(4) ≈ 1.386294
    logits = torch.tensor([[1.38629436, 0.0]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)

    brier = brier_score_loss(logits, target).item()
    # (0.8 - 1)^2 + (0.2 - 0)^2 = 0.04 + 0.04 = 0.08
    assert pytest.approx(brier, abs=1e-4) == 0.08

    # 4 classes uniform: logits [0, 0, 0, 0] -> p = 0.25
    # Brier = (0.25-1)^2 + 3 * (0.25)^2 = 0.5625 + 3 * 0.0625 = 0.75 = 1 - 1/4
    logits_uniform = torch.zeros((1, 4), dtype=torch.float32)
    target_uniform = torch.tensor([2], dtype=torch.long)
    brier_uniform = brier_score_loss(logits_uniform, target_uniform).item()
    assert pytest.approx(brier_uniform, abs=1e-5) == 0.75


def test_brier_gradient_matches_analytic_derivation() -> None:
    # Check that autograd gradient matches:
    # d(B)/dz_j = 2 * p_j * [ (p_j - y_j) - sum_k p_k * (p_k - y_k) ]
    logits = torch.tensor([[2.5, -1.0, 0.5, 1.2]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0], dtype=torch.long)

    loss = brier_score_loss(logits, target)
    loss.backward()
    numeric_grad = logits.grad.clone()

    probs = F.softmax(logits.detach(), dim=-1)[0]
    one_hot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    diff = probs - one_hot
    inner_term = torch.sum(probs * diff)
    analytic_grad = 2.0 * probs * (diff - inner_term)

    assert torch.allclose(numeric_grad[0], analytic_grad, atol=1e-5)


def test_temperature_calibrator_optimization_and_argmax() -> None:
    torch.manual_seed(42)
    # Generate overconfident synthetic logits (scaled by 5.0)
    base_logits = torch.randn((200, 5)) * 5.0
    targets = torch.randint(0, 5, (200,))

    calibrator = TemperatureCalibrator(model_id="test-bert", initial_temperature=1.0)
    initial_loss = cross_entropy_loss(calibrator(base_logits), targets).item()

    calibrator.fit(base_logits, targets, max_iter=50)
    fitted_temp = calibrator.temperature
    calibrated_loss = cross_entropy_loss(calibrator(base_logits), targets).item()

    # Loss after calibration should be strictly lower
    assert calibrated_loss <= initial_loss
    assert fitted_temp > 0.0

    # Temperature scaling must strictly preserve argmax
    raw_argmax = base_logits.argmax(dim=-1)
    scaled_argmax = calibrator(base_logits).argmax(dim=-1)
    assert torch.equal(raw_argmax, scaled_argmax)


def test_compute_ece_equal_width_and_mass() -> None:
    # Perfectly calibrated synthetic predictions:
    # 100 samples at conf 0.8 with 80% accuracy, 100 samples at conf 0.4 with 40% accuracy
    np.random.seed(42)
    confs = np.array([0.8] * 100 + [0.4] * 100)
    accs = np.array([1] * 80 + [0] * 20 + [1] * 40 + [0] * 60)

    res_width = compute_ece(confs, accs, n_bins=10, strategy="equal_width")
    assert pytest.approx(res_width.ece, abs=1e-4) == 0.0

    res_mass = compute_ece(confs, accs, n_bins=2, strategy="equal_mass")
    assert pytest.approx(res_mass.ece, abs=1e-4) == 0.0

    # Overconfident: all conf 0.9, but accuracy is only 0.5
    confs_bad = np.array([0.9] * 100)
    accs_bad = np.array([1] * 50 + [0] * 50)
    res_bad = compute_ece(confs_bad, accs_bad, n_bins=10, strategy="equal_width")
    assert pytest.approx(res_bad.ece, abs=1e-4) == 0.4
