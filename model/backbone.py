from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROUTE_TOKEN = "<|route|>"


def load_decoder(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def load_encoder(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


class FrozenDecoderBackbone:
    def __init__(self, model, tokenizer, route_token: str = ROUTE_TOKEN, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.route_token = route_token
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _append_route(self, ids: list[int]) -> list[int]:
        return ids + self.tokenizer.encode(self.route_token, add_special_tokens=False)

    @torch.no_grad()
    def route_hidden(self, text: str, max_length: int = 512) -> torch.Tensor:
        route_ids = self.tokenizer.encode(self.route_token, add_special_tokens=False)
        budget = max(1, max_length - len(route_ids))
        ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=budget)
        ids = self._append_route(ids)
        input_ids = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[-1][:, -1, :]


class FrozenEncoderBackbone:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def route_hidden(self, text: str, max_length: int = 512) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled


from model.router import ACTIONS, TERMINALS, TOOLS

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOLS)}
TERMINAL_TO_IDX = {t: i for i, t in enumerate(TERMINALS)}


def record_text(rec: dict[str, Any]) -> str:
    return rec["user_request"] + "\n" + rec.get("repository_summary", "")


def extract_bundle(
    records: list[dict[str, Any]],
    hidden_fn: Callable[[str], torch.Tensor],
    desc: str = "extract",
) -> dict[str, Any]:
    from tqdm import tqdm

    if not records:
        raise ValueError("no records")
    feats, labels_action, labels_tool, labels_terminal, labels_conf, ids = [], [], [], [], [], []
    for rec in tqdm(records, desc=desc):
        h = hidden_fn(record_text(rec))
        feats.append(h.squeeze(0).cpu().float())
        action = rec["label_action"]
        if action not in ACTION_TO_IDX:
            raise ValueError(f"record {rec.get('id', '?')} unknown label_action {action!r}")
        labels_action.append(ACTION_TO_IDX[action])
        if action == "TOOL":
            tool = rec.get("label_tool")
            if tool not in TOOL_TO_IDX:
                raise ValueError(
                    f"record {rec['id']} TOOL missing/unknown label_tool {tool!r}"
                )
            labels_tool.append(TOOL_TO_IDX[tool])
        else:
            labels_tool.append(-1)
        terminal = rec.get("label_terminal")
        if terminal not in TERMINAL_TO_IDX:
            raise ValueError(
                f"record {rec.get('id', '?')} unknown label_terminal {terminal!r}"
            )
        labels_terminal.append(TERMINAL_TO_IDX[terminal])
        conf = rec.get("label_confidence", 0.5)
        try:
            labels_conf.append(float(conf))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {rec.get('id', '?')} bad label_confidence {conf!r}") from exc
        ids.append(rec["id"])
    return {
        "ids": ids,
        "features": torch.stack(feats),
        "action": torch.tensor(labels_action, dtype=torch.long),
        "tool": torch.tensor(labels_tool, dtype=torch.long),
        "terminal": torch.tensor(labels_terminal, dtype=torch.long),
        "confidence": torch.tensor(labels_conf, dtype=torch.float32),
        "hard_negative": torch.tensor([bool(r.get("hard_negative", False)) for r in records]),
    }
