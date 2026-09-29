import json

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from model.router import AdaptiveRouter
from scripts.training.train_lora import (
    load_or_attach_adapter,
    load_or_create_router,
    train_lora_on_model,
)


def tiny_setup():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def sample_shard(tmp_path):
    path = tmp_path / "shard.jsonl"
    lines = []
    for i in range(8):
        lines.append(json.dumps({
            "id": f"t{i}",
            "user_request": "Fix the test.",
            "repository_summary": "repo",
            "target_text": "def fix(): return True",
            "label_action": "TOOL",
            "label_tool": "RUN_TESTS",
            "label_terminal": "CONTINUE",
            "label_confidence": 0.8,
        }))
    path.write_text("\n".join(lines))
    return path


def test_train_lora_smoke(tmp_path):
    model, tok = tiny_setup()
    result = train_lora_on_model(
        model=model,
        tokenizer=tok,
        shard_path=sample_shard(tmp_path),
        out_dir=tmp_path / "out",
        lora_cfg={"r": 4, "alpha": 8, "dropout": 0.05,
                  "target_modules": ["q_proj", "v_proj"]},
        router_cfg={"hidden_size": 32, "mlp_hidden": 16, "mlp_out": 8},
        loss_cfg={"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
                  "lambda_conf": 0.5, "lambda_budget": 0.0},
        train_cfg={"lr": 1e-3, "epochs": 1, "micro_batch_size": 2,
                   "gradient_accumulation": 1, "max_length": 64},
        seed=0,
        device="cpu",
    )
    assert result["steps"] > 0
    assert (tmp_path / "out" / "adapter_config.json").exists()
    assert (tmp_path / "out" / "router.pt").exists()
    assert (tmp_path / "out" / "router_config.json").exists()


ROUTER_CFG = {"hidden_size": 32, "mlp_hidden": 16, "mlp_out": 8}


def lora_cfg():
    return LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="CAUSAL_LM")


def test_load_or_create_router_fresh(tmp_path):
    router = load_or_create_router(tmp_path, ROUTER_CFG)
    assert isinstance(router, AdaptiveRouter)
    assert not (tmp_path / "router.pt").exists()


def test_load_or_create_router_resumes(tmp_path):
    ref = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    torch.save(ref.state_dict(), tmp_path / "router.pt")
    router = load_or_create_router(tmp_path, ROUTER_CFG)
    ref_state, got_state = ref.state_dict(), router.state_dict()
    assert ref_state.keys() == got_state.keys()
    for k in ref_state:
        assert torch.equal(ref_state[k], got_state[k])


def test_load_or_create_router_shape_mismatch_falls_back(tmp_path):
    torch.save(AdaptiveRouter(hidden_size=64, mlp_hidden=16, mlp_out=8).state_dict(),
               tmp_path / "router.pt")
    router = load_or_create_router(tmp_path, ROUTER_CFG)
    fresh = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    for (ka, va), (kf, vf) in zip(router.state_dict().items(), fresh.state_dict().items()):
        assert ka == kf and va.shape == vf.shape


def test_load_or_attach_adapter_fresh(tmp_path):
    model, _tok = tiny_setup()
    wrapped = load_or_attach_adapter(model, lora_cfg(), tmp_path)
    assert isinstance(wrapped, PeftModel)
    assert not (tmp_path / "adapter_config.json").exists()


def test_load_or_attach_adapter_resumes(tmp_path):
    model, _tok = tiny_setup()
    first = load_or_attach_adapter(model, lora_cfg(), tmp_path)
    first.save_pretrained(tmp_path)
    assert (tmp_path / "adapter_config.json").exists()
    model2, _tok2 = tiny_setup()
    resumed = load_or_attach_adapter(model2, lora_cfg(), tmp_path)
    assert isinstance(resumed, PeftModel)
    w1 = {n: p.detach().clone() for n, p in first.named_parameters() if "lora_" in n}
    w2 = {n: p.detach() for n, p in resumed.named_parameters() if "lora_" in n}
    assert w1 and w1.keys() == w2.keys()
    k = next(iter(w1))
    assert torch.equal(w1[k], w2[k])