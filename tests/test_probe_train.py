from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from model.backbone import ROUTE_TOKEN, FrozenDecoderBackbone, extract_bundle
from scripts.data_generation.generate_decisions import generate_dataset
from scripts.training.train_router import train_router


def tiny_bundle(n=48):
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    records = generate_dataset(n=n, seed=0)
    return extract_bundle(records, bb.route_hidden, desc="test-extract"), 32


def _coeffs():
    return {"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
            "lambda_conf": 0.5, "lambda_budget": 0.0}


def test_train_router_smoke(tmp_path):
    bundle, hidden = tiny_bundle()
    result = train_router(
        bundle, hidden_size=hidden, epochs=3, batch_size=16, lr=1e-3,
        coeffs=_coeffs(), out_dir=tmp_path, seed=0,
    )
    assert result["final_loss"] < result["first_loss"]
    assert (tmp_path / "router.pt").exists()
    assert (tmp_path / "router_config.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_train_router_overfits_tiny_batch(tmp_path):
    bundle, _hidden = tiny_bundle(n=12)
    metrics = train_router(
        bundle, hidden_size=32, epochs=30, batch_size=12, lr=1e-2,
        coeffs=_coeffs(), out_dir=tmp_path, seed=0,
    )
    assert metrics["final_loss"] < metrics["first_loss"] * 0.1


def test_train_router_seed_reproducible(tmp_path):
    bundle, hidden = tiny_bundle()
    r1 = train_router(bundle, hidden, epochs=1, batch_size=16, lr=1e-3,
                      coeffs=_coeffs(), out_dir=tmp_path / "a", seed=7)
    r2 = train_router(bundle, hidden, epochs=1, batch_size=16, lr=1e-3,
                      coeffs=_coeffs(), out_dir=tmp_path / "b", seed=7)
    assert abs(r1["final_loss"] - r2["final_loss"]) < 1e-6
