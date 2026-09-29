from __future__ import annotations

import torch


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = 15,
) -> float:
    if confidence.shape != correct.shape:
        raise ValueError(
            f"confidence and correct must share shape, got {tuple(confidence.shape)} vs {tuple(correct.shape)}"
        )
    confidence = confidence.detach().float().cpu()
    correct = correct.detach().float().cpu()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = confidence.numel()
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if mask.any():
            acc = correct[mask].mean().item()
            conf = confidence[mask].mean().item()
            ece += (mask.sum().item() / n) * abs(acc - conf)
    return float(ece)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 0.01,
    steps: int = 200,
) -> float:
    logits = logits.detach().float().cpu()
    labels = labels.detach().cpu().long()
    raw = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([raw], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / torch.exp(raw), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(raw.detach()).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-6)
