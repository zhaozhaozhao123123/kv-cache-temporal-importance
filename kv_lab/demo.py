"""Deterministic SYNTHETIC Q/K/V experiment, not a pretrained-model benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import attention, importance, protection, quantize_cache, relative_error
from .report import csv_write, export_trace


def run(out, seed=42, steps=48, prompt=64, heads=2, dim=32, alpha=0.2, window=8):
    if steps < 2 or prompt < 8:
        raise ValueError("Need at least 2 steps and 8 prompt tokens")
    rng = np.random.default_rng(seed)
    total = prompt + steps
    keys = rng.normal(size=(heads, total, dim))
    keys[:, :, 3] *= 4  # Persistent key channel outlier.
    values = rng.normal(size=keys.shape)
    values[:, ::11, 7] *= 10  # Scattered value outliers.
    trace = np.full((steps, total), np.nan)
    snapshots = []
    for s in range(steps):
        n = prompt + s + 1
        anchor = (5, 23, 47)[min(2, 3 * s // steps)] % prompt
        q = keys[:, anchor, :].copy() * 1.5 + rng.normal(scale=0.1, size=(heads, dim))
        # Suppress channel outlier in queries; query relevance changes by stage.
        q[:, 3] *= 0.08
        k, v = keys[:, :n], values[:, :n]
        weights, result = attention(q, k, v)
        trace[s, :n] = weights.mean(axis=0)
        snapshots.append((q, k, v, result))
    scores = export_trace(trace, prompt, out, dict(source="SYNTHETIC_QKV", seed=seed,
        heads=heads, head_dim=dim, changing_anchors=[5, 23, 47],
        limitations="Float quantize/dequantize reference only; no packed memory, speed or LLM-quality claims"),
        alpha=alpha, window=window)
    rows = []
    for s, (q, k, v, ref) in enumerate(snapshots):
        n = k.shape[1]
        for bits in (2, 4):
            for policy in ("recent", "cumulative", "window", "ema", "random", "ema_outliers"):
                method = "ema" if policy == "ema_outliers" else policy
                history = np.full(n, np.nan)
                if s and method in scores:
                    history[:n - 1] = scores[method][s - 1, :n - 1]
                elif method == "random":
                    history[:max(0, n - 8)] = np.random.default_rng(seed + s).random(max(0, n - 8))
                protected = protection(history, n, recent=8, budget=0 if policy == "recent" else 8)
                ratio = 0.02 if policy == "ema_outliers" else 0
                kq = quantize_cache(k, "key", bits, protected=protected, outlier_fraction=ratio)
                vq = quantize_cache(v, "value", bits, protected=protected, outlier_fraction=ratio)
                _, result = attention(q, kq, vq)
                rows.append(dict(step=s, bits=bits, policy=policy, protected_tokens=len(protected),
                    key_mse=float(np.mean((k - kq) ** 2)), value_mse=float(np.mean((v - vq) ** 2)),
                    relative_output_error=relative_error(ref, result)))
    csv_write(Path(out) / "quantization.csv", rows)
    summary = []
    for bits in (2, 4):
        for policy in dict.fromkeys(r["policy"] for r in rows):
            # Skip step 0: temporal policies do not yet have historical evidence.
            selected = [r for r in rows if r["bits"] == bits and r["policy"] == policy and r["step"] > 0]
            summary.append(dict(bits=bits, policy=policy, steps=len(selected),
                mean_relative_output_error=float(np.mean([r["relative_output_error"] for r in selected])),
                mean_protected_tokens=float(np.mean([r["protected_tokens"] for r in selected]))))
    csv_write(Path(out) / "summary.csv", summary)
    (Path(out) / "README_RESULTS.md").write_text(
        "# Synthetic experiment results\n\nThese results describe seeded synthetic Q/K/V, not Qwen or other pretrained models.\n"
        "Attention is softmax(QK/sqrt(D))V. Quantization is an offline float simulation.\n"
        "Temporal protection uses previous-step scores. The recent-only baseline has fewer full-precision tokens;\n"
        "compare cumulative/window/EMA/random at equal token budget. EMA+outliers uses extra sparse storage,\n"
        "so it is not a storage-matched comparison. 2% is a requested fraction; ceil per group can exceed it.\n"
        "No measured memory compression, latency, perplexity or task accuracy is reported.\n",
        encoding="utf-8")
    print(json.dumps(dict(source="SYNTHETIC_QKV", steps=steps, configurations=12, rows=len(rows), out=str(out)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--window", type=int, default=8)
    args = p.parse_args()
    run(args.out, args.seed, args.steps, alpha=args.alpha, window=args.window)


if __name__ == "__main__":
    main()
