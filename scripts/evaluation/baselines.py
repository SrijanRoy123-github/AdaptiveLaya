from __future__ import annotations

import re
from typing import Any

import torch

TOOL_KEYWORDS = re.compile(
    r"\b(run|test|lint|find|search|grep|fix the failing|"
    r"simulate|synthesize|route|drill|layout|spice|verify|place|draw)\b",
    re.IGNORECASE,
)
ASK_KEYWORDS = re.compile(
    r"\b(better|improve|clean|vague|something|stuff|explain|summarize|"
    r"what is|how does|describe|review)\b",
    re.IGNORECASE,
)
MATH_KEYWORDS = re.compile(r"\b(calculate|compute|solve|prove|integrate|derivative|equation)\b", re.IGNORECASE)
SCIENCE_KEYWORDS = re.compile(r"\b(experiment|hypothesis|measure|observe|analyze|formula|physics|chemistry|biology)\b", re.IGNORECASE)


def always_answer(bundle: dict[str, Any]) -> dict[str, torch.Tensor]:
    n = bundle["features"].shape[0]
    return {
        "action": torch.zeros(n, dtype=torch.long),
        "tool": torch.full((n,), -1, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }


def always_tool(bundle: dict[str, Any]) -> dict[str, torch.Tensor]:
    n = bundle["features"].shape[0]
    return {
        "action": torch.ones(n, dtype=torch.long),
        "tool": torch.zeros(n, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }


def keyword_router(prompts: list[str]) -> dict[str, torch.Tensor]:
    actions, tools = [], []
    for p in prompts:
        if MATH_KEYWORDS.search(p):
            actions.append(1)
            tools.append(10)
        elif SCIENCE_KEYWORDS.search(p):
            actions.append(1)
            tools.append(13)
        elif TOOL_KEYWORDS.search(p):
            actions.append(1)
            tools.append(0)
        elif ASK_KEYWORDS.search(p):
            actions.append(2)
            tools.append(-1)
        else:
            actions.append(0)
            tools.append(-1)
    n = len(prompts)
    return {
        "action": torch.tensor(actions, dtype=torch.long),
        "tool": torch.tensor(tools, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }
