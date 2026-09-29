from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset

from model.losses import router_loss
from model.router import ACTIONS, TERMINALS, TOOLS, AdaptiveRouter
from scripts.training.train_router import set_seed

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOLS)}
TERMINAL_TO_IDX = {t: i for i, t in enumerate(TERMINALS)}


class DecisionSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int = 256):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        prompt = rec["user_request"] + "\n" + rec.get("repository_summary", "")
        target = rec.get("target_text", "") or f"<|{rec['label_action'].lower()}>"
        enc = self.tok(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        tgt = self.tok(
            target, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        input_ids = torch.cat([enc["input_ids"], tgt["input_ids"]], dim=1).squeeze(0)
        attn = torch.cat([enc["attention_mask"], tgt["attention_mask"]], dim=1).squeeze(0)
        labels = input_ids.clone()
        labels[: enc["input_ids"].shape[1]] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "action": ACTION_TO_IDX[rec["label_action"]],
            "tool": TOOL_TO_IDX.get(rec.get("label_tool") or "RUN_TESTS", -1),
            "terminal": TERMINAL_TO_IDX.get(rec.get("label_terminal", "CONTINUE"), 0),
            "confidence": float(rec.get("label_confidence", 0.5)),
        }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    out = {
        "input_ids": torch.full((len(batch), max_len), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), max_len), dtype=torch.long),
        "labels": torch.full((len(batch), max_len), -100, dtype=torch.long),
        "action": torch.tensor([b["action"] for b in batch], dtype=torch.long),
        "tool": torch.tensor([b["tool"] for b in batch], dtype=torch.long),
        "terminal": torch.tensor([b["terminal"] for b in batch], dtype=torch.long),
        "confidence": torch.tensor([b["confidence"] for b in batch], dtype=torch.float32),
    }
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        out["input_ids"][i, :L] = item["input_ids"]
        out["attention_mask"][i, :L] = item["attention_mask"]
        out["labels"][i, :L] = item["labels"]
    return out


def load_or_create_router(out_dir: str | Path, router_cfg: dict[str, Any], device: str = "cpu") -> AdaptiveRouter:
    router = AdaptiveRouter(
        hidden_size=int(router_cfg["hidden_size"]),
        mlp_hidden=int(router_cfg.get("mlp_hidden", 512)),
        mlp_out=int(router_cfg.get("mlp_out", 256)),
    )
    router_path = Path(out_dir) / "router.pt"
    if router_path.exists():
        try:
            router.load_state_dict(torch.load(router_path, map_location="cpu", weights_only=True))
            print(f"[resume] router loaded from {router_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - shape/config drift must not kill a long run
            print(f"[resume] router init skipped ({exc}); training fresh", flush=True)
    return router.to(device)


def load_or_attach_adapter(model, lora_cfg: LoraConfig, out_dir: str | Path):
    if (Path(out_dir) / "adapter_config.json").exists():
        try:
            wrapped = PeftModel.from_pretrained(model, str(out_dir))
            print(f"[resume] adapters loaded from {out_dir}", flush=True)
            return wrapped
        except Exception as exc:  # noqa: BLE001 - corrupt/partial adapter must not kill a long run
            print(f"[resume] adapter init skipped ({exc}); training fresh", flush=True)
    return get_peft_model(model, lora_cfg)


def train_lora_on_model(
    model,
    tokenizer,
    shard_path: str | Path,
    out_dir: str | Path,
    lora_cfg: dict[str, Any],
    router_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int = 42,
    device: str = "cpu",
    hours_limit: float | None = None,
) -> dict[str, Any]:
    import time

    set_seed(seed)
    records = [json.loads(line) for line in Path(shard_path).read_text(encoding="utf-8").splitlines()]

    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = load_or_attach_adapter(model, lora_config, out_dir)
    model.to(device)

    router = load_or_create_router(out_dir, router_cfg, device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ds = DecisionSFTDataset(records, tokenizer, max_length=int(train_cfg.get("max_length", 256)))
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg.get("micro_batch_size", 1)),
        shuffle=True,
        collate_fn=lambda b: _collate(b, pad_id),
    )

    params = list(model.parameters()) + list(router.parameters())
    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 2e-4)))
    epochs = int(train_cfg.get("epochs", 1))
    accum = int(train_cfg.get("gradient_accumulation", 1))
    # fp16 backbone + fp32 LoRA params mismatch at LayerNorm without autocast
    # ("expected scalar type Half but found Float"); AMP keeps the forward in
    # mixed precision with fp32 master grads. No-op on CPU / fp16: false.
    use_amp = bool(train_cfg.get("fp16", True)) and str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch <2.3 fallback
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]
    started = time.monotonic()
    steps = 0
    last_loss = None

    model.train()
    router.train()
    for _ in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels, output_hidden_states=True)
                lm_loss = out.loss
                hidden = out.hidden_states[-1][:, -1, :]
            lm_loss = lm_loss.float()
            hidden = hidden.float()
            router_out = router(hidden, torch.full((batch["input_ids"].shape[0], 4), 0.5, device=device))
            labels_router = {
                "action": batch["action"].to(device),
                "tool": batch["tool"].to(device),
                "terminal": batch["terminal"].to(device),
                "confidence": batch["confidence"].to(device),
            }
            r_parts = router_loss(router_out, labels_router, loss_cfg)
            loss = (lm_loss + r_parts["loss"]) / accum
            scaler.scale(loss).backward()
            last_loss = float(loss.item())
            if (steps + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            steps += 1
            if hours_limit is not None and (time.monotonic() - started) / 3600.0 >= hours_limit:
                break

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    torch.save(router.state_dict(), out_path / "router.pt")
    (out_path / "router_config.json").write_text(json.dumps(router_cfg, indent=2))
    return {"steps": steps, "last_loss": last_loss, "examples": len(records)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/phase2_lora.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import yaml

    from model.backbone import load_decoder

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model, tok = load_decoder(cfg["backbone"]["model_id"], device=args.device)
    metrics = train_lora_on_model(
        model, tok, args.shard, args.out,
        lora_cfg=cfg["lora"], router_cfg=cfg["router"], loss_cfg=cfg["loss"],
        train_cfg=cfg["training"], seed=args.seed, device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()