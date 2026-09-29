from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.routing_heads import ConfidenceHead, Head

ACTIONS = ["ANSWER", "TOOL", "ASK"]
TOOLS = [
    "RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY",
    "VLSI_SPICE", "VLSI_SYNTHESIZE", "VLSI_LAYOUT", "VLSI_ROUTE",
    "CAD_DRC", "CAD_LAYOUT",
    "MATH_COMPUTE", "REASON_DEDUCE", "BOOK_SUMMARIZE", "SCIENCE_QUERY",
]
TERMINALS = ["CONTINUE", "DONE", "ESCALATE"]


@dataclass
class RouterOutput:
    action_logits: torch.Tensor
    tool_logits: torch.Tensor
    terminal_logits: torch.Tensor
    confidence: torch.Tensor


class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 512, mlp_out: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_out),
        )
        self.budget_proj = nn.Linear(4, mlp_out)
        self.action_head = Head(mlp_out, len(ACTIONS))
        self.tool_head = Head(mlp_out, len(TOOLS))
        self.terminal_head = Head(mlp_out, len(TERMINALS))
        self.confidence_head = ConfidenceHead(mlp_out)

    def forward(self, h: torch.Tensor, budget: torch.Tensor | None = None) -> RouterOutput:
        if budget is not None and budget.shape[-1] != 4:
            raise ValueError(f"budget must have 4 dims, got shape {tuple(budget.shape)}")
        x = self.mlp(self.norm(h))
        if budget is not None:
            x = x + self.budget_proj(budget)
        return RouterOutput(
            action_logits=self.action_head(x),
            tool_logits=self.tool_head(x),
            terminal_logits=self.terminal_head(x),
            confidence=self.confidence_head(x),
        )

    @torch.no_grad()
    def predict(self, h: torch.Tensor, budget: torch.Tensor | None = None) -> dict:
        self.eval()
        if h.dim() != 2 or h.shape[0] != 1:
            raise ValueError(f"predict expects a single sample, got shape {tuple(h.shape)}")
        out = self.forward(h, budget)
        action_idx = int(out.action_logits.argmax(-1).item())
        action = ACTIONS[action_idx]
        tool = None
        if action == "TOOL":
            tool = TOOLS[int(out.tool_logits.argmax(-1).item())]
        return {
            "action": action,
            "tool": tool,
            "terminal": TERMINALS[int(out.terminal_logits.argmax(-1).item())],
            "confidence": float(out.confidence.item()),
        }
