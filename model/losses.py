from __future__ import annotations

import torch
import torch.nn.functional as F

from model.router import ACTIONS, RouterOutput


def router_loss(
    out: RouterOutput,
    labels: dict[str, torch.Tensor],
    coeffs: dict[str, float],
) -> dict[str, torch.Tensor]:
    action = labels["action"]
    tool = labels["tool"]
    terminal = labels["terminal"]
    confidence = labels["confidence"]

    action_loss = F.cross_entropy(out.action_logits, action)
    tool_mask = (action == ACTIONS.index("TOOL")) & (tool >= 0)
    if tool_mask.any():
        tool_loss = F.cross_entropy(out.tool_logits[tool_mask], tool[tool_mask])
    else:
        tool_loss = out.tool_logits.sum() * 0.0
    terminal_loss = F.cross_entropy(out.terminal_logits, terminal)
    conf_loss = F.mse_loss(out.confidence, confidence)

    loss = (
        float(coeffs.get("lambda_action", 1.0)) * action_loss
        + float(coeffs.get("lambda_tool", 1.0)) * tool_loss
        + float(coeffs.get("lambda_terminal", 1.0)) * terminal_loss
        + float(coeffs.get("lambda_conf", 1.0)) * conf_loss
    )
    return {
        "loss": loss,
        "action_loss": action_loss.detach(),
        "tool_loss": tool_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "conf_loss": conf_loss.detach(),
    }


def utility_weighted_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cost_weights: torch.Tensor,
) -> torch.Tensor:
    per = F.cross_entropy(logits, targets, reduction="none")
    return (per * cost_weights).mean()
