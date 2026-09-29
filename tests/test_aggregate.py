import json

import torch

from scripts.aggregate_adapters import (
    aggregate_checkpoints,
    average_router_states,
    merge_full_states,
    select_best_worker,
    validate_pair,
)
from scripts.verify_checkpoint import verify_checkpoint


def state(v):
    return {"layer.weight": torch.full((4, 4), float(v))}


def test_router_average_equal():
    a, b, c = state(1.0), state(3.0), state(5.0)
    avg = average_router_states([a, b, c])
    assert torch.allclose(avg["layer.weight"], torch.full((4, 4), 3.0))


def test_router_average_weighted():
    avg = average_router_states([state(0.0), state(10.0)], weights=[1, 3])
    assert torch.allclose(avg["layer.weight"], torch.full((4, 4), 7.5))


def test_merge_full_states_delta_average():
    base = state(0.0)
    w1, w2 = state(2.0), state(4.0)
    merged = merge_full_states(base, [w1, w2])
    assert torch.allclose(merged["layer.weight"], torch.full((4, 4), 3.0))


def test_validate_pair_rejects_shape_mismatch():
    ok, msg = validate_pair(state(1), state(2))
    assert ok is True
    bad = {"layer.weight": torch.zeros(2, 2)}
    ok, msg = validate_pair(state(1), bad)
    assert ok is False
    assert "shape" in msg


def test_validate_pair_rejects_nan():
    nan_state = {"layer.weight": torch.tensor([[float("nan")]])}
    ok, _msg = validate_pair(state(1), nan_state)
    assert ok is False


def test_validate_pair_rejects_nan_in_ref():
    nan_state = {"layer.weight": torch.full((4, 4), float("nan"))}
    ok, msg = validate_pair(nan_state, state(1))
    assert ok is False
    assert "non-finite" in msg


def test_select_best_worker():
    workers = [
        {"worker_id": 0, "validation_score": 0.70},
        {"worker_id": 1, "validation_score": 0.75},
        {"worker_id": 2, "validation_score": 0.72},
    ]
    best = select_best_worker(workers)
    assert best["worker_id"] == 1


def _worker_dir(tmp_path, name, worker_id, val_loss, v):
    d = tmp_path / name
    d.mkdir()
    torch.save(state(v), d / "router.pt")
    (d / "router_config.json").write_text(json.dumps({"hidden_size": 4}))
    (d / "manifest.json").write_text(json.dumps({
        "worker_id": worker_id, "round": 0, "seed": 42 + worker_id, "step": 10,
        "base_revision": "rev-a", "val_loss": val_loss,
    }))
    return d


def test_aggregate_then_verify_merged_ok(tmp_path):
    w0 = _worker_dir(tmp_path, "agg0", 0, 0.020, 1.0)
    w1 = _worker_dir(tmp_path, "agg1", 1, 0.030, 3.0)
    out = tmp_path / "merged"
    manifest = aggregate_checkpoints([w0, w1], "validation_weighted", out)
    assert manifest["method"] == "validation_weighted"
    assert len(manifest["weights"]) == 2 and abs(sum(manifest["weights"]) - 1.0) < 1e-6
    assert (out / "router.pt").exists()
    result = verify_checkpoint(out)
    assert result["ok"] is True, result["problems"]
