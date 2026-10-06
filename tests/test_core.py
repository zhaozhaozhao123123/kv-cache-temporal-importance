import numpy as np
import pytest
from kv_lab.core import (attention, expand_gqa, importance, topk_jaccard,
                         quantize_cache, protection, block_mass)


def test_attention_matches_hand_computed_softmax():
    q = np.array([[1., 0.]])
    k = np.array([[[1., 0.], [0., 1.]]])
    v = np.array([[[2., 0.], [0., 4.]]])
    a, y = attention(q, k, v)
    p = np.exp(1 / np.sqrt(2)) / (1 + np.exp(1 / np.sqrt(2)))
    np.testing.assert_allclose(a, [[p, 1-p]])
    np.testing.assert_allclose(y, [[2*p, 4*(1-p)]])


def test_new_token_is_not_diluted_by_prebirth_zeros():
    x = np.array([[.9, np.nan], [.3, .7]])
    assert importance(x, "ema", alpha=.5)[1, 1] == .7
    assert importance(x, "window", window=2)[1, 1] == .7
    assert importance(x, "ema", alpha=.5)[1, 0] == pytest.approx(.6)


def test_no_future_attention_leakage():
    x = np.array([[.8, .2], [.1, .9], [.7, .3]])
    for m in ("instant", "cumulative", "ema", "window"):
        np.testing.assert_allclose(importance(x, m)[:2], importance(x[:2], m))


def test_key_per_channel_preserves_channel_constants():
    # Each key channel is constant across time; per-token quantization would distort it.
    x = np.tile(np.array([0., .1, 10., 100.]), (1, 7, 1))
    np.testing.assert_array_equal(quantize_cache(x, "key", 2), x)


def test_value_per_token_preserves_token_constants():
    x = np.array([[[1., 1., 1.], [100., 100., 100.], [-4., -4., -4.]]])
    np.testing.assert_array_equal(quantize_cache(x, "value", 2), x)


@pytest.mark.parametrize("kind", ["key", "value"])
def test_guard_excluded_from_ranges_and_preserved(kind):
    x = np.arange(60, dtype=float).reshape(1, 5, 12)
    original = x.copy()
    x[:, 0] = 1e9
    y = quantize_cache(x, kind, 2, protected=[0])
    y2 = quantize_cache(original, kind, 2, protected=[0])
    np.testing.assert_array_equal(y[:, 0], x[:, 0])
    np.testing.assert_allclose(y[:, 1:], y2[:, 1:])
    assert np.isfinite(y).all()


def test_sparse_outlier_keeps_extreme_value():
    x = np.array([[[0., .1, .2, 1000.]]])
    y = quantize_cache(x, "value", 2, outlier_fraction=.25)
    assert y[0, 0, -1] == 1000
    assert abs(y[0, 0, 1] - .1) <= .04


def test_protection_has_disjoint_budgets_and_uses_supplied_history():
    ids = protection(np.array([.1, .9, .2, .3]), 5, recent=2, budget=1)
    np.testing.assert_array_equal(ids, [1, 3, 4])


def test_common_cohort_jaccard_ignores_new_tokens():
    a = np.array([.9, .1, np.nan])
    b = np.array([.1, .9, 100.])
    assert topk_jaccard(a, b, 1) == 0


def test_logical_block_mass_conserves_total():
    scores = np.array([.1, .2, .3, .4, np.nan])
    assert block_mass(scores, 2).sum() == pytest.approx(1)


def test_gqa_head_order():
    cache = np.array([[[1., 2.]], [[3., 4.]]])
    np.testing.assert_array_equal(expand_gqa(cache, 4)[:, 0, 0], [1, 1, 3, 3])


def test_invalid_bit_width_rejected():
    with pytest.raises(ValueError):
        quantize_cache(np.ones((1, 4, 4)), "key", bits=1)
