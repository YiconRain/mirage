#!/usr/bin/env python3
"""Plot Fig. 13 — compute–communication overlap ablation.

Reads the per-cell JSONs emitted by run_fig13_overlap.sh under
results/H100/fig13/ and produces a grouped bar chart matching the
paper figure (light blue = without overlap, orange = with overlap),
with per-pair speedup annotations.

Usage:
    python artifact_evaluation/H100/plot_fig13.py
    python artifact_evaluation/H100/plot_fig13.py --input-dir <dir> --output <png>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_input = here.parents[2] / "results" / "H100" / "fig13"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=default_input,
                   help=f"Directory of mpk-*.json files (default: {default_input})")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <input-dir>/fig13.png)")
    p.add_argument("--gpu-label", type=str, default=None,
                   help="Override GPU label in the title (default: read from JSON)")
    return p.parse_args()


def load_cells(input_dir: Path) -> dict[str, dict[int, float]]:
    """Return {mode: {batch_size: per_iteration_us}}."""
    if not input_dir.is_dir():
        sys.exit(f"input dir not found: {input_dir}")
    out: dict[str, dict[int, float]] = {"overlap": {}, "no-overlap": {}}
    for path in sorted(input_dir.glob("mpk-*__bs*.json")):
        with path.open() as f:
            cell = json.load(f)
        mode = cell.get("mode")
        bs = int(cell.get("batch_size"))
        lat_ms = float(cell.get("latency_ms_per_token"))
        us = lat_ms * 1000.0
        if mode not in out:
            out[mode] = {}
        out[mode][bs] = us
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    cells = load_cells(args.input_dir)
    overlap = cells.get("overlap", {})
    no_overlap = cells.get("no-overlap", {})
    batch_sizes = sorted(set(overlap) & set(no_overlap))
    if not batch_sizes:
        sys.exit(f"no paired (overlap, no-overlap) cells in {args.input_dir}")
    missing_o = sorted(set(no_overlap) - set(overlap))
    missing_n = sorted(set(overlap) - set(no_overlap))
    if missing_o or missing_n:
        print(f"warning: unpaired cells skipped (no overlap match: {missing_o}, "
              f"no no-overlap match: {missing_n})", file=sys.stderr)

    no_vals = np.array([no_overlap[bs] for bs in batch_sizes])
    ov_vals = np.array([overlap[bs] for bs in batch_sizes])
    speedups = no_vals / ov_vals

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(batch_sizes))
    width = 0.38
    color_no = "#a8c6e5"
    color_ov = "#f0a060"
    ax.bar(x - width / 2, no_vals, width, label="MPK (without overlap)",
           color=color_no, edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, ov_vals, width, label="MPK (with overlap)",
           color=color_ov, edgecolor="black", linewidth=0.4)

    ymax = max(no_vals.max(), ov_vals.max())
    ax.set_ylim(0, ymax * 1.25)
    for i, sp in enumerate(speedups):
        ax.text(x[i], max(no_vals[i], ov_vals[i]) + ymax * 0.05,
                f"{sp:.1f}x", ha="center", va="bottom",
                color=color_ov, fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={b}" for b in batch_sizes])
    ax.set_ylabel("Per-iteration Runtime (us)")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    output = args.output or (args.input_dir / "fig13.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"saved {output}")

    print("\nbs   without (us)   with (us)   speedup")
    for bs, n, o, sp in zip(batch_sizes, no_vals, ov_vals, speedups):
        print(f"{bs:>3}   {n:>12.1f}   {o:>9.1f}   {sp:>5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
