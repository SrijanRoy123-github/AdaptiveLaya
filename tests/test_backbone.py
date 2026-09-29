import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    Qwen2Config,
)

from model.backbone import ROUTE_TOKEN, FrozenDecoderBackbone, FrozenEncoderBackbone


def tiny_model_and_tokenizer():
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


def test_route_hidden_shape():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    h = bb.route_hidden("def add(a, b): return a + b")
    assert h.shape[-1] == 32
    assert h.ndim == 2


def test_route_hidden_deterministic_and_frozen():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    assert model.training is False
    h1 = bb.route_hidden("x = 1")
    h2 = bb.route_hidden("x = 1")
    assert torch.allclose(h1, h2, atol=1e-5)
    assert all(not p.requires_grad for p in model.parameters())


def test_route_token_appended():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    ids = tok.encode("hi", add_special_tokens=False)
    appended = bb._append_route(ids)
    route_ids = tok.encode(ROUTE_TOKEN, add_special_tokens=False)
    assert appended == ids + route_ids


def test_encoder_route_hidden_shape_and_frozen(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "def", "foo", "(", ":", ")", "pass"]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab_file))
    cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=128,
    )
    model = AutoModel.from_config(cfg)
    bb = FrozenEncoderBackbone(model, tok, device="cpu")
    h = bb.route_hidden("def foo(): pass")
    assert h.shape == (1, 32)
    assert all(not p.requires_grad for p in model.parameters())
    assert model.training is False
