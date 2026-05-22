#!/usr/bin/env python3
"""Plot Fig. 11 — Qwen3-1.7B multi-GPU comparison across systems.

Reads results/H100/fig11/<system>__tp<TP>__bs<BS>.json files and produces
a grouped bar chart with one subplot per TP value. Y-axis is relative
performance, normalized so MPK = 1.0 at each (TP, BS). Speedups above the
MPK bar show MPK / next-best system, matching the paper convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_ORDER = ["pytorch", "vllm", "sglang", "mpk"]
SYSTEM_LABEL = {
    "pytorch": "PyTorch",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "mpk": "MPK",
}
SYSTEM_COLOR = {
    "pytorch": "#2c4f9f",
    "vllm": "#8cc97a",
    "sglang": "#a9bee2",
    "mpk": "#f0a060",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_input = here.parents[2] / "results" / "H100" / "fig11"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=default_input,
                   help=f"Directory of <system>__tp<TP>__bs<BS>.json files "
                        f"(default: {default_input})")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <input-dir>/fig11.png)")
    return p.parse_args()


def load_cells(input_dir: Path) -> dict:
    """Return {tp: {bs: {system: latency_ms_per_token}}}."""
    if not input_dir.is_dir():
        sys.exit(f"input dir not found: {input_dir}")
    cells: dict = {}
    for path in sorted(input_dir.glob("*__tp*__bs*.json")):
        with path.open() as f:
            c = json.load(f)
        system = c.get("system")
        tp = int(c.get("tp"))
        bs = int(c.get("batch_size"))
        lat = float(c.get("latency_ms_per_token"))
        cells.setdefault(tp, {}).setdefault(bs, {})[system] = lat
    return cells


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    cells = load_cells(args.input_dir)
    tps = sorted(cells.keys())
    if not tps:
        sys.exit(f"no cells found in {args.input_dir}")

    fig, axes = plt.subplots(1, len(tps), figsize=(4.5 * len(tps), 4.2),
                              sharey=True)
    if len(tps) == 1:
        axes = [axes]

    for ax, tp in zip(axes, tps):
        bss = sorted(cells[tp].keys())
        x = np.arange(len(bss))
        width = 0.2

        mpk_vals = []
        per_system = {s: [] for s in SYSTEM_ORDER}
        for bs in bss:
            row = cells[tp][bs]
            mpk_lat = row.get("mpk")
            mpk_vals.append(mpk_lat)
            for s in SYSTEM_ORDER:
                lat = row.get(s)
                # relative perf = mpk_lat / system_lat (higher is better;
                # MPK normalized to 1.0). Missing cells become NaN.
                if lat is None or mpk_lat is None or lat == 0:
                    per_system[s].append(float("nan"))
                else:
                    per_system[s].append(mpk_lat / lat)

        for i, s in enumerate(SYSTEM_ORDER):
            offset = (i - (len(SYSTEM_ORDER) - 1) / 2) * width
            ax.bar(x + offset, per_system[s], width,
                   label=SYSTEM_LABEL[s],
                   color=SYSTEM_COLOR[s],
                   edgecolor="black", linewidth=0.4)

        # Speedup of MPK over the next-best non-MPK system (the one with
        # the highest relative perf among {pytorch, vllm, sglang}).
        for i, bs in enumerate(bss):
            others = [per_system[s][i] for s in SYSTEM_ORDER if s != "mpk"]
            others = [v for v in others if v == v]  # drop NaN
            if not others:
                continue
            best_other = max(others)
            speedup = 1.0 / best_other if best_other > 0 else float("nan")
            ax.text(x[i], 1.05, f"{speedup:.1f}x",
                    ha="center", va="bottom",
                    color=SYSTEM_COLOR["mpk"], fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"BS={b}" for b in bss], fontsize=9)
        ax.set_xlabel(f"{tp} GPUs", fontsize=11)
        ax.set_ylim(0, 1.25)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("Relative Performance", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(SYSTEM_ORDER), frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output = args.output or (args.input_dir / "fig11.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"saved {output}")

    print("\nlatency_ms_per_token by (TP, BS, system):")
    for tp in tps:
        for bs in sorted(cells[tp].keys()):
            row = cells[tp][bs]
            cells_str = "  ".join(f"{s}={row.get(s, '-')}" for s in SYSTEM_ORDER)
            print(f"  tp={tp} bs={bs}: {cells_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
