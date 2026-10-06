import csv
import json
import platform
from pathlib import Path
import numpy as np
from .core import importance, topk_jaccard, block_mass


def csv_write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_trace(trace, prompt_len, out, metadata, labels=None, alpha=0.2, window=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "attention_trace.npz", attention=trace, prompt_len=prompt_len)
    scores = {m: importance(trace, m, alpha, window)
              for m in ("instant", "cumulative", "ema", "window")}
    rows, dynamics, blocks = [], [], []
    for s in range(len(trace)):
        for t in np.flatnonzero(np.isfinite(trace[s])):
            rows.append(dict(step=s, token_position=int(t), label=(labels[t] if labels else f"token_{t}"),
                             **{m: float(a[s, t]) for m, a in scores.items()}))
        for name, a in scores.items():
            dynamics.append(dict(step=s, method=name, prompt_top8_jaccard=(
                topk_jaccard(a[s - 1, :prompt_len], a[s, :prompt_len], 8) if s else "")))
        for b, mass in enumerate(block_mass(trace[s])):
            blocks.append(dict(step=s, block=b, block_size=16, attention_mass=float(mass)))
    csv_write(out / "token_scores.csv", rows)
    csv_write(out / "rank_dynamics.csv", dynamics)
    csv_write(out / "logical_blocks.csv", blocks)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    im = axes[0].imshow(np.ma.masked_invalid(trace.T), aspect="auto", origin="lower", cmap="viridis")
    axes[0].set(xlabel="Decode query step", ylabel="KV token position", title=f"{metadata['source']} | attention trace")
    fig.colorbar(im, ax=axes[0], label="Mean attention probability")
    selected = np.argsort(-np.nansum(trace[:, :prompt_len], axis=0))[:4]
    for t in selected:
        axes[1].plot(scores["ema"][:, t], label=f"prompt position {t}")
    axes[1].set(xlabel="Decode query step", ylabel="EMA importance", title="Temporal importance of selected prompt tokens")
    axes[1].legend()
    fig.savefig(out / "importance.png", dpi=180)
    plt.close(fig)
    metadata.update(python=platform.python_version(), numpy=np.__version__, prompt_tokens=prompt_len,
                    steps=len(trace), alpha=alpha, window=window,
                    score_definition="Mean attention over selected layers and query heads; not causal importance",
                    rank_cohort="Prompt tokens only, top-8 set Jaccard")
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores
