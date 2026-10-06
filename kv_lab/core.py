"""Numerical reference implementations; no packed storage or GPU kernels."""
import numpy as np


def attention(query, keys, values):
    """Q [H,D], K/V [H,T,D]; attention probability [H,T]."""
    if keys.shape != values.shape or query.shape != (keys.shape[0], keys.shape[2]):
        raise ValueError("Expected Q[H,D] and matching K,V[H,T,D]")
    logits = np.einsum("hd,htd->ht", query, keys) / np.sqrt(keys.shape[2])
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights, np.einsum("ht,htd->hd", weights, values)


def expand_gqa(cache, query_heads):
    """Repeat each KV head for its query-head group, as in grouped-query attention."""
    if query_heads % cache.shape[0]:
        raise ValueError("Query-head count must be divisible by KV-head count")
    return np.repeat(cache, query_heads // cache.shape[0], axis=0)


def importance(trace, method="ema", alpha=0.2, window=8):
    """Trace[S,T] uses NaN for not-yet-visible tokens, never zero padding.

    EMA initializes each token at its first observation. Window mean divides by
    that token's actual visible observations. Scores are NOT causal effect estimates.
    """
    x = np.asarray(trace, dtype=float)
    if x.ndim != 2 or len(x) == 0 or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
        raise ValueError("Expected nonnegative [steps,tokens] trace with NaN visibility mask")
    if not 0 < alpha <= 1 or window < 1:
        raise ValueError("Invalid EMA alpha or window")
    if method == "instant":
        return x.copy()
    if method == "cumulative":
        out = np.cumsum(np.nan_to_num(x), axis=0)
        out[~np.isfinite(x)] = np.nan
        return out
    out = np.full_like(x, np.nan)
    if method == "ema":
        state = np.full(x.shape[1], np.nan)
        for s, row in enumerate(x):
            valid = np.isfinite(row)
            born = valid & ~np.isfinite(state)
            old = valid & np.isfinite(state)
            state[born] = row[born]
            state[old] = alpha * row[old] + (1 - alpha) * state[old]
            out[s, valid] = state[valid]
        return out
    if method == "window":
        for s in range(len(x)):
            rows = x[max(0, s - window + 1):s + 1]
            count = np.isfinite(rows).sum(axis=0)
            out[s] = np.divide(np.nansum(rows, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0)
            out[s, ~np.isfinite(x[s])] = np.nan
        return out
    raise ValueError(f"Unknown importance method: {method}")


def topk(scores, k):
    """Stable ranking: ties are resolved by ascending token position."""
    ids = np.flatnonzero(np.isfinite(scores))
    return ids[np.argsort(-np.asarray(scores)[ids], kind="stable")[:k]]


def topk_jaccard(a, b, k):
    common = np.isfinite(a) & np.isfinite(b)
    a = np.where(common, a, np.nan)
    b = np.where(common, b, np.nan)
    sa, sb = set(topk(a, k)), set(topk(b, k))
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def block_mass(scores, block_size=16):
    """Logical token-block aggregation, NOT PagedAttention memory allocation."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return np.array([np.nansum(scores[t:t + block_size])
                     for t in range(0, len(scores), block_size)])


def _uniform_group(x, bits, outlier_fraction):
    """Group affine quantize/dequantize, optionally preserving large magnitudes."""
    sparse = np.zeros(x.shape, dtype=bool)
    count = min(len(x) - 1, int(np.ceil(len(x) * outlier_fraction)))
    if count:
        sparse[np.argsort(-np.abs(x), kind="stable")[:count]] = True
    dense = ~sparse
    lo, hi = x[dense].min(), x[dense].max()
    levels = 2 ** bits - 1
    if hi == lo:
        y = np.full_like(x, lo)
    else:
        scale = (hi - lo) / levels
        codes = np.clip(np.rint((x - lo) / scale), 0, levels)
        y = codes * scale + lo
    y[sparse] = x[sparse]
    return y


def quantize_cache(x, kind, bits=2, group_size=16, protected=(), outlier_fraction=0):
    """K: group along time for EACH channel; V: group channels for EACH token.

    Protected tokens are removed BEFORE estimating group ranges and restored
    exactly, so their outliers cannot enlarge the low-bit quantization range.
    Returned floats are simulated dequantized values, not compressed tensors.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all() or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("Expected finite cache [heads,tokens,channels]")
    if kind not in {"key", "value"} or bits not in {2, 3, 4, 8} or group_size < 1:
        raise ValueError("Invalid quantization settings")
    if not 0 <= outlier_fraction < 1:
        raise ValueError("Outlier fraction must be in [0,1)")
    guard = np.zeros(x.shape[1], dtype=bool)
    ids = np.asarray(list(protected), dtype=int)
    if np.any(ids < 0) or np.any(ids >= x.shape[1]):
        raise ValueError("Protected token position outside cache")
    guard[ids] = True
    y = x.copy()
    for h in range(x.shape[0]):
        if kind == "key":
            for start in range(0, x.shape[1], group_size):
                tids = np.arange(start, min(start + group_size, x.shape[1]))
                tids = tids[~guard[tids]]
                if not len(tids):
                    continue
                for d in range(x.shape[2]):
                    y[h, tids, d] = _uniform_group(x[h, tids, d], bits, outlier_fraction)
        else:
            for t in np.flatnonzero(~guard):
                for start in range(0, x.shape[2], group_size):
                    sl = slice(start, start + group_size)
                    y[h, t, sl] = _uniform_group(x[h, t, sl], bits, outlier_fraction)
    return y


def protection(scores, tokens, recent=8, budget=8):
    """Pick old tokens using PREVIOUS-step scores; recent and old budgets are disjoint."""
    if recent < 0 or budget < 0 or len(scores) > tokens:
        raise ValueError("Invalid protection budget or score length")
    start = max(0, tokens - recent)
    old = np.full(tokens, np.nan)
    old[:len(scores)] = scores
    old[start:] = np.nan
    return np.union1d(np.arange(start, tokens), topk(old, budget))


def relative_error(reference, candidate):
    return float(np.linalg.norm(reference - candidate) /
                 max(np.linalg.norm(reference), 1e-12))
