from __future__ import annotations

import copy
import random
from typing import Any

if __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.data_generation.generate_decisions import build_branches

HARD_TEMPLATES = [
    {
        "prompt": "Run the tests and fix the failure.",
        "label_action": "ANSWER",
        "note": "sounds like TOOL but missing repo context, answer with clarifying plan",
    },
    {
        "prompt": "What does this function return?",
        "label_action": "TOOL",
        "note": "question-like but requires reading context to answer precisely",
    },
    {
        "prompt": "Fix everything that is broken.",
        "label_action": "ASK",
        "note": "underspecified scope requires clarification",
    },
    {
        "prompt": "Add type hints across the package.",
        "label_action": "ASK",
        "note": "ambiguous target package",
    },
]


def add_hard_negatives(records: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = [copy.deepcopy(r) for r in records]
    if not out:
        raise ValueError("records must be non-empty")
    start = len(out)
    for j in range(n):
        tmpl = HARD_TEMPLATES[j % len(HARD_TEMPLATES)]
        rec = copy.deepcopy(out[j % len(out)])
        rec["id"] = f"hn-{start + j:06d}"
        rec["prompt"] = tmpl["prompt"]
        rec["user_request"] = tmpl["prompt"]
        rec["label_action"] = tmpl["label_action"]
        rec["label_tool"] = None if tmpl["label_action"] != "TOOL" else "READ_CONTEXT"
        rec["hard_negative"] = True
        rec["note"] = tmpl["note"]
        rec["label_confidence"] = round(rng.uniform(0.5, 0.85), 3)
        rec["branches"] = build_branches(rec["label_action"], rng)
        out.append(rec)
    rng.shuffle(out)
    return out
