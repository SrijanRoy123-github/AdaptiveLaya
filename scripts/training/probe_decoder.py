import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import torch

from model.backbone import FrozenDecoderBackbone, extract_bundle, load_decoder


def _resolve_device_dtype(device: str):
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _validate_device(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(2)


def save_bundle_atomic(bundle, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not Path(args.input).exists():
        ap.error(f"input not found: {args.input}")
    _validate_device(args.device)
    dtype = _resolve_device_dtype(args.device)
    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, tok = load_decoder(args.model_id, device=args.device, dtype=dtype)
    print(f"model loaded; extracting {len(records)} records", flush=True)
    backbone = FrozenDecoderBackbone(model, tok, device=args.device)
    bundle = extract_bundle(records, backbone.route_hidden, desc="extract-decoder")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle_atomic(bundle, out)
    print(f"saved {bundle['features'].shape} to {out}", flush=True)


if __name__ == "__main__":
    main()
