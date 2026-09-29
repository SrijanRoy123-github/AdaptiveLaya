import json

from scripts.data_generation.counterfactuals import label_counterfactuals
from scripts.data_generation.generate_decisions import generate_dataset, write_dataset
from scripts.data_generation.hard_negatives import add_hard_negatives
from scripts.data_generation.perturb import perturb_numbers, rename_identifiers

REGIMES = {
    "LOW": {"success_reward": 1.0, "alpha_tokens": 2e-4, "beta_tool_calls": 5e-2, "gamma_latency_ms": 1e-5},
    "NORMAL": {"success_reward": 1.0, "alpha_tokens": 1e-4, "beta_tool_calls": 2e-2, "gamma_latency_ms": 5e-6},
    "HIGH": {"success_reward": 1.0, "alpha_tokens": 5e-5, "beta_tool_calls": 1e-2, "gamma_latency_ms": 2e-6},
}


def test_generate_balanced_ids_unique():
    data = generate_dataset(n=300, seed=0)
    assert len(data) == 300
    ids = [d["id"] for d in data]
    assert len(set(ids)) == 300
    assert {d["label_action"] for d in data} == {"ANSWER", "TOOL", "ASK"}


def test_write_and_reload(tmp_path):
    data = generate_dataset(n=50, seed=1)
    out = write_dataset(data, tmp_path)
    assert out.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["count"] == 50
    assert manifest["sha256"]
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 50


def test_hard_negatives_flagged():
    data = generate_dataset(n=200, seed=2)
    hn = add_hard_negatives(data, n=40, seed=3)
    flagged = [d for d in hn if d["hard_negative"]]
    assert len(flagged) >= 40
    assert all(d["label_action"] in {"ANSWER", "TOOL", "ASK"} for d in hn)


def test_counterfactuals_multi_regime_labels():
    data = generate_dataset(n=60, seed=4)
    labeled = label_counterfactuals(data, REGIMES)
    assert set(labeled[0]["regime_best_action"]) == {"LOW", "NORMAL", "HIGH"}
    for d in labeled:
        for v in d["regime_best_action"].values():
            assert v in {"ANSWER", "TOOL", "ASK"}


def test_perturb_helpers():
    original = "timeout = 30 seconds, limit = 100"
    s = perturb_numbers(original, seed=1)
    assert s != original
    s2 = rename_identifiers("def compute_total(items):")
    assert "compute_total" not in s2


def test_manifest_checksum_matches_on_disk_bytes(tmp_path):
    import hashlib

    data = generate_dataset(n=20, seed=5)
    write_dataset(data, tmp_path)
    data_bytes = (tmp_path / "decisions.jsonl").read_bytes()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(data_bytes).hexdigest()


def test_generate_dataset_deterministic():
    a = generate_dataset(n=60, seed=3)
    b = generate_dataset(n=60, seed=3)
    assert a == b


def test_action_difficulty_decorrelated():
    recs = generate_dataset(n=300, seed=0)
    cells = {(r["label_action"], r["difficulty"]) for r in recs}
    assert len(cells) == 9


def test_counterfactual_regime_patterns_vary():
    recs = generate_dataset(n=100, seed=0)
    labeled = label_counterfactuals(recs, REGIMES)
    patterns = {
        tuple(r["regime_best_action"][k] for k in ("LOW", "NORMAL", "HIGH"))
        for r in labeled
    }
    assert len(patterns) >= 2
    agree = sum(1 for r in labeled if r["regime_best_action"]["NORMAL"] == r["label_action"])
    assert agree / len(labeled) >= 0.9


def test_hard_negative_label_consistent():
    recs = generate_dataset(n=90, seed=0)
    hn = add_hard_negatives(recs, n=12, seed=1)
    checked = 0
    for r in hn:
        if r.get("hard_negative") and isinstance(r.get("branches"), dict) and "_chosen" in r["branches"]:
            assert r["branches"]["_chosen"]["action"] == r["label_action"]
            checked += 1
    assert checked == 12


def test_rename_identifiers_deterministic():
    t = "def compute_total(total): return total + ref_1"
    assert rename_identifiers(t, seed=5) == rename_identifiers(t, seed=5)
    # Whole-token rename: "total" inside "compute_total" must never produce a
    # "compute_ref_*" hybrid (substring corruption); compute_total is renamed
    # atomically to one ref (consistent with plan test_perturb_helpers).
    out = rename_identifiers("let total = compute_total(total)", seed=5)
    assert "compute_ref" not in out
    assert "compute_total" not in out
