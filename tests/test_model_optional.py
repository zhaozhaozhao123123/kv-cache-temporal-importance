"""No downloads: a tiny RANDOM Qwen3 validates cache-growth plumbing only."""
import numpy as np
import pytest


def test_tiny_random_qwen_cached_vs_full_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kv_lab.collect_qwen import collect
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
           head_dim=8, max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    trace, n, tokens, layers = collect(model, ids, 3, [-1])
    assert trace.shape == (3, 7) and n == 4 and layers == [1]
    np.testing.assert_allclose(np.nansum(trace, axis=1), 1, atol=1e-6)
    assert np.isnan(trace[0, 5:]).all()
    with torch.inference_mode():
        full = model(input_ids=torch.tensor([tokens]), output_attentions=True, use_cache=False)
    expected = full.attentions[-1][0, :, -1, :].mean(0).numpy()
    np.testing.assert_allclose(trace[-1], expected, rtol=1e-4, atol=1e-6)
