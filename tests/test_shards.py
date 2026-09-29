import hashlib
import json
import sys
from collections import Counter

from scripts.data_generation.generate_decisions import generate_dataset
from scripts.make_shards import main, shard_of, shard_records, split_shared_holdouts


def test_shard_of_stable_across_calls():
    assert shard_of("dec-000001", 3) == shard_of("dec-000001", 3)
    assert 0 <= shard_of("dec-000001", 3) < 3
    assert 0 <= shard_of("dec-000002", 3) < 3


def test_stratified_proportions():
    data = generate_dataset(n=900, seed=0)
    shards = shard_records(data, n=3, seed=42)
    for recs in shards:
        actions = Counter(r["label_action"] for r in recs)
        total = len(recs)
        for act in ("ANSWER", "TOOL", "ASK"):
            assert abs(actions[act] / total - 1 / 3) < 0.05


def test_holdouts_never_in_shards():
    data = generate_dataset(n=300, seed=1)
    val, test, _train = split_shared_holdouts(data, val_frac=0.1, test_frac=0.1, seed=0)
    train_ids = {r["id"] for r in _train}
    assert not train_ids & {r["id"] for r in val}
    assert not train_ids & {r["id"] for r in test}
    assert not {r["id"] for r in val} & {r["id"] for r in test}
    shards = shard_records(_train, n=3, seed=42)
    all_sharded = {r["id"] for s in shards for r in s}
    assert all_sharded == train_ids


def test_deterministic_across_runs():
    data = generate_dataset(n=120, seed=2)
    a = shard_records(data, n=3, seed=42)
    b = shard_records(data, n=3, seed=42)
    assert [[r["id"] for r in s] for s in a] == [[r["id"] for r in s] for s in b]


def test_permutation_invariance():
    recs = generate_dataset(n=90, seed=0)
    a = shard_records(recs, 3, seed=42)
    b = shard_records(list(reversed(recs)), 3, seed=42)
    assert [sorted(r["id"] for r in s) for s in a] == [sorted(r["id"] for r in s) for s in b]


def test_holdouts_stratified():
    recs = generate_dataset(n=300, seed=0)
    val, test, _train = split_shared_holdouts(recs, 0.1, 0.1, seed=0)
    for split in (val, test):
        c = Counter(r["label_action"] for r in split)
        total = sum(c.values())
        assert all(abs(v / total - 1 / 3) <= 0.10 for v in c.values())


def test_fraction_validation():
    recs = generate_dataset(n=30, seed=0)
    try:
        split_shared_holdouts(recs, 0.6, 0.5, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_manifest_written_and_matches(tmp_path, monkeypatch):
    recs = generate_dataset(n=60, seed=3)
    in_path = tmp_path / "in.jsonl"
    in_path.write_bytes(("\n".join(json.dumps(r) for r in recs) + "\n").encode("utf-8"))
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", ["make_shards.py", "--input", str(in_path), "--out", str(out)])
    main()
    manifest = json.loads((out / "shards.manifest.json").read_bytes())
    assert manifest["input_sha256"] == hashlib.sha256(in_path.read_bytes()).hexdigest()
    expected = {"val.jsonl", "test.jsonl", "shard-0.jsonl", "shard-1.jsonl", "shard-2.jsonl"}
    assert set(manifest["files"]) == expected
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    counts = manifest["counts"]
    assert counts["train"] == sum(counts["shards"])
    assert counts["val"] + counts["test"] + counts["train"] == 60
    assert len(list(out.iterdir())) == 6  # 5 outputs + manifest


def test_config_fractions_used(tmp_path, monkeypatch):
    import scripts.make_shards as ms

    cfg = tmp_path / "worker.yaml"
    cfg.write_bytes(b"shared_validation_fraction: 0.2\nshared_test_fraction: 0.3\n")
    monkeypatch.setattr(ms, "CONFIG_PATH", cfg)
    recs = generate_dataset(n=60, seed=4)
    in_path = tmp_path / "in.jsonl"
    in_path.write_bytes(("\n".join(json.dumps(r) for r in recs) + "\n").encode("utf-8"))
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", ["make_shards.py", "--input", str(in_path), "--out", str(out)])
    main()
    m = json.loads((out / "shards.manifest.json").read_bytes())
    assert m["val_frac"] == 0.2
    assert m["test_frac"] == 0.3
    assert m["counts"]["val"] == 12  # 20 per label * 0.2, 3 labels
    out2 = tmp_path / "shards2"
    monkeypatch.setattr(
        sys, "argv",
        ["make_shards.py", "--input", str(in_path), "--out", str(out2), "--val-frac", "0.05"],
    )
    main()
    m2 = json.loads((out2 / "shards.manifest.json").read_bytes())
    assert m2["val_frac"] == 0.05  # explicit CLI overrides config default
    assert m2["test_frac"] == 0.3
