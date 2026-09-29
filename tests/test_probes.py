import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model.backbone import (
    ACTION_TO_IDX,
    TERMINAL_TO_IDX,
    TOOL_TO_IDX,
    extract_bundle,
    record_text,
)
from model.router import ACTIONS, TERMINALS, TOOLS
from scripts.training import probe_decoder, probe_encoder
from scripts.training.probe_decoder import (
    _resolve_device_dtype,
    _validate_device,
    save_bundle_atomic,
)
from scripts.training.probe_decoder import main as decoder_main
from scripts.training.probe_encoder import main as encoder_main

ROOT = Path(__file__).resolve().parents[1]


def fake_hidden(text):
    return torch.zeros(1, 8)


def sample_records(n=3):
    return [
        {
            "id": f"r{i}",
            "label_action": ["ANSWER", "TOOL", "ASK"][i % 3],
            "label_tool": "RUN_TESTS" if i % 3 == 1 else None,
            "label_terminal": ["CONTINUE", "DONE", "ESCALATE"][i % 3],
            "label_confidence": 0.5 + 0.1 * i,
            "user_request": f"task {i}",
            "repository_summary": "repo",
            "hard_negative": i == 1,
        }
        for i in range(n)
    ]


def test_decoder_help_works_as_subprocess():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "training" / "probe_decoder.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_encoder_help_works_as_subprocess():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "training" / "probe_encoder.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_resolve_device_dtype_cpu():
    assert _resolve_device_dtype("cpu") == torch.float32


def test_resolve_device_dtype_cuda():
    assert _resolve_device_dtype("cuda") == torch.float16
    assert _resolve_device_dtype("cuda:1") == torch.float16


def test_validate_device_rejects_cuda_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        _validate_device("cuda")


def test_validate_device_allows_cpu(monkeypatch):
    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": lambda: False})())
    _validate_device("cpu")


def test_extract_bundle_missing_tool_label_raises():
    recs = sample_records(3)
    recs[1]["label_tool"] = None
    with pytest.raises(ValueError, match="label_tool"):
        extract_bundle(recs, fake_hidden)


def test_extract_bundle_unknown_tool_raises():
    recs = sample_records(3)
    recs[1]["label_tool"] = "UNKNOWN_TOOL"
    with pytest.raises(ValueError, match="UNKNOWN_TOOL"):
        extract_bundle(recs, fake_hidden)


def test_extract_bundle_non_tool_missing_tool_is_minus_one():
    recs = sample_records(3)
    recs[0]["label_tool"] = None
    bundle = extract_bundle(recs, fake_hidden)
    assert bundle["tool"][0].item() == -1


def test_extract_bundle_empty_records_raises():
    with pytest.raises(ValueError, match="no records"):
        extract_bundle([], fake_hidden)


def test_extract_bundle_rejects_bad_action():
    recs = sample_records(3)
    recs[0]["label_action"] = "JUMP"
    with pytest.raises(ValueError, match="JUMP"):
        extract_bundle(recs, fake_hidden)


def test_extract_bundle_record_context_in_errors():
    recs = sample_records(3)
    recs[2]["id"] = "bad-id"
    recs[2]["label_action"] = "TOOL"
    recs[2]["label_tool"] = None
    with pytest.raises(ValueError, match="bad-id"):
        extract_bundle(recs, fake_hidden)


def test_record_text_composition():
    assert record_text({"user_request": "fix", "repository_summary": "api"}) == "fix\napi"


def test_label_maps_match_router_enums():
    assert ACTION_TO_IDX == {a: i for i, a in enumerate(ACTIONS)}
    assert TOOL_TO_IDX == {t: i for i, t in enumerate(TOOLS)}
    assert TERMINAL_TO_IDX == {t: i for i, t in enumerate(TERMINALS)}


def test_atomic_save_uses_tmp_then_replace(monkeypatch, tmp_path):
    out = tmp_path / "out.pt"
    calls = []

    def fake_save(bundle, path):
        calls.append(str(path))
        path.write_bytes(b"partial")

    monkeypatch.setattr(torch, "save", fake_save)
    replace_log = []

    def record_replace(self, target):
        replace_log.append((str(self), self.read_bytes()))
        target.unlink(missing_ok=True)

    monkeypatch.setattr(Path, "replace", record_replace)
    save_bundle_atomic({"x": 1}, out)
    assert not out.exists()
    assert calls == [str(tmp_path / "out.pt.tmp")]
    assert replace_log == [(str(tmp_path / "out.pt.tmp"), b"partial")]


def test_atomic_save_on_crash_does_not_corrupt(monkeypatch, tmp_path):
    out = tmp_path / "out.pt"

    def fake_save(bundle, path):
        if "tmp" in str(path):
            path.write_bytes(b"partial")
            raise RuntimeError("boom")

    monkeypatch.setattr(torch, "save", fake_save)
    with pytest.raises(RuntimeError, match="boom"):
        save_bundle_atomic({"x": 1}, out)
    assert not out.exists()


def test_decoder_desc_is_decoder_specific(monkeypatch, tmp_path, capsys):
    import json

    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps(sample_records()[0]))
    out = tmp_path / "out.pt"
    monkeypatch.setattr(sys, "argv", ["probe_decoder.py", "--input", str(inp), "--out", str(out)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class DummyBB:
        def route_hidden(self, text):
            return fake_hidden(text)

    monkeypatch.setattr(probe_decoder, "FrozenDecoderBackbone", lambda *a, **k: DummyBB())
    monkeypatch.setattr(probe_decoder, "load_decoder", lambda *a, **k: (object(), object()))
    decoder_main()
    assert "extract-decoder" in capsys.readouterr().err


def test_encoder_desc_is_encoder_specific(monkeypatch, tmp_path, capsys):
    import json

    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps(sample_records()[0]))
    out = tmp_path / "out.pt"
    monkeypatch.setattr(sys, "argv", ["probe_encoder.py", "--input", str(inp), "--out", str(out)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class DummyBB:
        def route_hidden(self, text):
            return fake_hidden(text)

    monkeypatch.setattr(probe_encoder, "FrozenEncoderBackbone", lambda *a, **k: DummyBB())
    monkeypatch.setattr(probe_encoder, "load_encoder", lambda *a, **k: (object(), object()))
    encoder_main()
    assert "extract-encoder" in capsys.readouterr().err
