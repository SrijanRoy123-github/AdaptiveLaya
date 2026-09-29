from __future__ import annotations

from scripts.data_generation.ingest_real import (
    _domain_from_text,
    _heuristic_label,
    local_mode,
)


def test_domain_detection():
    assert _domain_from_text("fix this verilog block with spice simulation") == "vlsi"
    assert _domain_from_text("optimize the kicad floorplan drc") == "cad"
    assert _domain_from_text("compute the integral of x squared") == "math"
    assert _domain_from_text("reason through this logic puzzle step by step") == "reasoning"
    assert _domain_from_text("summarize the chapter and characters") == "book"
    assert _domain_from_text("analyze the chemistry experiment results") == "science"
    assert _domain_from_text("write a python function to sort a list") == "coding"


def test_heuristic_label():
    labeled = _heuristic_label("simulate this spice netlist")
    assert labeled["label_action"] == "TOOL"
    assert labeled["label_tool"] == "VLSI_SPICE"
    assert labeled["domain"] == "vlsi"

    labeled = _heuristic_label("solve this math equation")
    assert labeled["label_action"] == "ANSWER"
    assert labeled["label_tool"] == "MATH_COMPUTE"


def test_local_mode_env(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_LAYA_LOCAL_DIR", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_SESSION", raising=False)
    assert local_mode() is True
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", "/tmp/store")
    assert local_mode() is True
