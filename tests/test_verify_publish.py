import json

import torch

from scripts.publish_checkpoint import build_manifest
from scripts.verify_checkpoint import verify_checkpoint


def make_ckpt(tmp_path, with_nan=False):
    root = tmp_path / "ckpt"
    root.mkdir()
    state = {"w": torch.ones(4)}
    if with_nan:
        state["w"][0] = float("nan")
    torch.save(state, root / "router.pt")
    (root / "router_config.json").write_text(json.dumps({"hidden_size": 8}))
    (root / "manifest.json").write_text(json.dumps(build_manifest(0, 0, 42, 5, "rev-a")))
    return root


def test_verify_ok(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path), expected_base_revision="rev-a")
    assert result["ok"] is True
    assert result["problems"] == []


def test_verify_missing_files(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    result = verify_checkpoint(root)
    assert result["ok"] is False


def test_verify_base_revision_mismatch(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path), expected_base_revision="rev-b")
    assert result["ok"] is False
    assert any("base_revision" in p for p in result["problems"])


def test_verify_nan_detected(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path, with_nan=True))
    assert result["ok"] is False


def test_build_manifest_fields():
    m = build_manifest(2, 1, 44, 100, "rev-c", extra={"mode": "probe"})
    assert m["worker_id"] == 2 and m["round"] == 1 and m["seed"] == 44
    assert m["step"] == 100 and m["base_revision"] == "rev-c"
    assert m["mode"] == "probe"
    assert m["created_utc"]


def test_verify_adapter_not_lora(tmp_path):
    root = make_ckpt(tmp_path)
    (root / "adapter_config.json").write_text(json.dumps({"peft_type": "ADALORA"}))
    result = verify_checkpoint(root)
    assert result["ok"] is False
    assert any("adapter" in p for p in result["problems"])


def test_verify_merged_manifest_ok(tmp_path):
    root = make_ckpt(tmp_path)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    m.pop("worker_id")
    m.update({"method": "validation_weighted", "weights": [0.5, 0.5],
              "workers": ["results/agg/0", "results/agg/1"]})
    (root / "manifest.json").write_text(json.dumps(m))
    result = verify_checkpoint(root)
    assert result["ok"] is True, result["problems"]


def test_verify_merged_empty_workers_fails(tmp_path):
    root = make_ckpt(tmp_path)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    m.pop("worker_id")
    m["workers"] = []
    (root / "manifest.json").write_text(json.dumps(m))
    result = verify_checkpoint(root)
    assert result["ok"] is False
    assert any("workers" in p for p in result["problems"])
