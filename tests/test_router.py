import torch

from model.router import ACTIONS, TERMINALS, TOOLS, AdaptiveRouter


def test_output_shapes():
    router = AdaptiveRouter(hidden_size=64, mlp_hidden=32, mlp_out=16)
    h = torch.randn(4, 64)
    budget = torch.rand(4, 4)
    out = router(h, budget)
    assert out.action_logits.shape == (4, len(ACTIONS))
    assert out.tool_logits.shape == (4, len(TOOLS))
    assert out.terminal_logits.shape == (4, len(TERMINALS))
    assert out.confidence.shape == (4,)
    assert torch.all((out.confidence >= 0) & (out.confidence <= 1))


def test_param_count_half_m_to_5m():
    router = AdaptiveRouter(hidden_size=896, mlp_hidden=512, mlp_out=256)
    n = sum(p.numel() for p in router.parameters())
    assert 500_000 <= n <= 5_000_000


def test_predict_action_masks_tool_when_not_tool():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    decision = router.predict(torch.randn(1, 32), torch.rand(1, 4))
    assert decision["action"] in ACTIONS
    if decision["action"] != "TOOL":
        assert decision["tool"] is None
    assert 0.0 <= decision["confidence"] <= 1.0
    assert decision["terminal"] in TERMINALS


def test_predict_tool_branch():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    with torch.no_grad():
        router.action_head.proj.bias[ACTIONS.index("TOOL")] = 10.0
    decision = router.predict(torch.randn(1, 32), torch.zeros(1, 4))
    assert decision["action"] == "TOOL"
    assert decision["tool"] in TOOLS


def test_predict_rejects_batch():
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    try:
        router.predict(torch.randn(4, 32), torch.zeros(4, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_forward_budget_none_path():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    out = router(torch.randn(32), None)
    assert out.action_logits.shape == (3,)
    # budget_proj receives no gradient on this path (documented footgun)
    for p in router.budget_proj.parameters():
        assert p.grad is None


def test_constants_match_action_schema():
    import json
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "action_schema.json").read_text(encoding="utf-8"))
    action_enum = schema["properties"]["action"]["enum"]
    tool_enum = [t for t in schema["properties"]["tool"]["enum"] if t is not None]
    terminal_enum = schema["properties"]["terminal"]["enum"]
    assert list(ACTIONS) == action_enum
    assert list(TOOLS) == tool_enum
    assert list(TERMINALS) == terminal_enum
