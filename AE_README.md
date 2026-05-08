# Artifact Evaluation — TGX (OSDI '26 paper #804)

*TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*

The system is called **MPK** in this codebase and **TGX** in the paper; the
two names refer to the same artifact. This file documents how to reproduce
each figure of the paper's evaluation (§6).

> **Branch.** `tgx-osdi26-ae` — the frozen artifact for OSDI '26 AE.

---

## Evaluation matrix

| Exp. | Paper figure | Models                | GPU(s)        | Batch sizes        |
|------|--------------|-----------------------|---------------|--------------------|
| E1   | Fig. 9 (§6.3) | 5 (Qwen3-{0.6B,1.7B,8B,30B-A3B}, Llama-3.2-1B) | A100 / H100 / B200 | 1, 2, 4, 8, 16 |
| E2   | Fig. 10 (§6.4) | Qwen3-30B-A3B (MoE)  | B200          | 1, 2, 4, 8, 16     |
| E3   | Fig. 11 (§6.5) | Qwen3-1.7B           | 2/4/8 × H100  | 1, 2, 4, 8, 16     |
| E4   | Fig. 12 (§6.6) | Qwen3-8B (final linear only) | B200 | 1, 2, 4, 8, 16     |
| E5   | Fig. 13 (§6.6) | Qwen3-1.7B           | 4 × H100      | 1, 2, 4, 8, 16     |

**Workload (all experiments).** Offline batched inference, prompt length
**64**, decode **1024** tokens, greedy (`--temperature 0`). Numbers in the
paper are the median of 5 runs after a 4-iteration warmup.

**Metric.** Per-token decoding latency (ms) printed by every demo:

```
Prompt length 64, generate length 1024, per-token latency (both prefill and decode): X.XXX ms
```

Convert to throughput as `tokens/sec = 1000 / latency_ms`. All bar charts in
the paper are normalized to TGX's throughput.

**TGX configuration (Table 1).** Set automatically per GPU type:

| GPU  | # SMs | # workers | # schedulers |
|------|-------|-----------|--------------|
| A100 | 108   | 104       | 16           |
| H100 | 132   | 128       | 16           |
| B200 | 148   | 144       | 16           |

Shared-memory page = 32 KB.

**Models.**

| Paper name           | HuggingFace ID                | Demo entry point             |
|----------------------|-------------------------------|------------------------------|
| Qwen3-0.6B           | `Qwen/Qwen3-0.6B`             | `demo/qwen3/demo.py`         |
| Llama-3.2-1B-Instruct| `meta-llama/Llama-3.2-1B-Instruct` | `demo/llama3/demo.py`   |
| Qwen3-1.7B           | `Qwen/Qwen3-1.7B`             | `demo/qwen3/demo.py`         |
| Qwen3-8B             | `Qwen/Qwen3-8B`               | `demo/qwen3/demo.py`         |
| Qwen3-30B-A3B        | `Qwen/Qwen3-30B-A3B`          | `demo/qwen3/demo_30B_A3B.py` |

**Baselines.** PyTorch (= the demos run *without* `--use-mirage`), vLLM, and
SGLang. Wrapper scripts under `scripts/ae/baselines/` produce JSON output in
the same format as the TGX driver, so plotting is uniform.

---

## E1 — Single-GPU end-to-end (Fig. 9, §6.3)

**Claim.** TGX reduces per-token decoding latency by **1.0×–1.7×** vs. the
best of {PyTorch, vLLM, SGLang} across 5 models × 3 GPUs × 5 batch sizes.
Qwen3-30B-A3B on A100 is omitted (OOM).

**Driver.**

```bash
# gpu ∈ {a100, h100, b200}
bash scripts/ae/run_e1.sh <gpu>
```

This loops over (model, batch_size, system) and writes
`results/ae/e1/<gpu>/<model>__<system>.json`. Plot with:

```bash
python scripts/ae/plot_fig9.py --root results/ae/e1 --out fig9.pdf
```

**Single-cell command (TGX).**

```bash
python demo/qwen3/demo.py \
    --use-mirage --model Qwen/Qwen3-8B \
    --max-num-batched-requests <bs> \
    --max-new-tokens 1024 --temperature 0
```

For Llama-3.2-1B-Instruct, swap to `demo/llama3/demo.py` with
`--model meta-llama/Llama-3.2-1B-Instruct`.

**Single-cell command (PyTorch).** Same as above, drop `--use-mirage`.

**Single-cell command (vLLM).**

```bash
bash scripts/ae/baselines/run_vllm.sh <hf-id> <bs>
```

**Single-cell command (SGLang).**

```bash
bash scripts/ae/baselines/run_sglang.sh <hf-id> <bs>
```

**Expected runtime.** ~12 GPU-hours per GPU type for the full matrix.

---

## E2 — MoE case study (Fig. 10, §6.4)

**Claim.** TGX-Hybrid-MoE outperforms SGLang-MoE on Qwen3-30B-A3B (B200)
by **1.07×–1.18×** across batch sizes; TGX-Hybrid also beats TGX-Static
because the latter over-statically partitions experts.

**Driver.**

```bash
bash scripts/ae/run_e2.sh
python scripts/ae/plot_fig10.py --root results/ae/e2 --out fig10.pdf
```

**Single-cell commands.**

```bash
# TGX-Hybrid (default)
python demo/qwen3/demo_30B_A3B.py --use-mirage \
    --max-num-batched-requests <bs> \
    --max-new-tokens 1024 --temperature 0

# TGX-Static (force static expert partitioning)
python demo/qwen3/demo_30B_A3B.py --use-mirage --moe-static \
    --max-num-batched-requests <bs> \
    --max-new-tokens 1024 --temperature 0

# SGLang-MoE baseline
bash scripts/ae/baselines/run_sglang.sh Qwen/Qwen3-30B-A3B <bs>
```

The metric in Fig. 10 is per-iteration **MoE-layer** runtime in microseconds,
isolated by the demo's profiler; raw timing is dumped to the JSON output.

---

## E3 — Multi-GPU tensor parallelism (Fig. 11, §6.5)

**Claim.** Under TP on 2/4/8 × H100, TGX outperforms vLLM and SGLang by
**1.0×–1.4×** on Qwen3-1.7B across batch sizes.

**Prereq.** Cache the model locally once (multi-GPU shard loader requires a
local checkpoint):

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /tmp/qwen3-1.7b
```

**Driver.**

```bash
bash scripts/ae/run_e3.sh <N>     # N ∈ {2, 4, 8}
python scripts/ae/plot_fig11.py --root results/ae/e3 --out fig11.pdf
```

**Single-cell command (TGX).**

```bash
mpirun -n <N> --allow-run-as-root \
    python demo/qwen3/demo.py --use-mirage \
        --model Qwen/Qwen3-1.7B --model-path /tmp/qwen3-1.7b \
        --max-num-batched-requests <bs> \
        --max-new-tokens 1024 --temperature 0
```

**Baselines.**

```bash
bash scripts/ae/baselines/run_vllm.sh   Qwen/Qwen3-1.7B <bs> --tp <N>
bash scripts/ae/baselines/run_sglang.sh Qwen/Qwen3-1.7B <bs> --tp <N>
```

---

## E4 — Ablation: cross-task pipelining (Fig. 12, §6.6)

**Claim.** Cross-task pipelining (paged shared memory + load/compute split,
§5.3) reduces final-linear runtime in Qwen3-8B on B200 by **1.2×–1.3×** vs.
TGX without pipelining, and outperforms a cuBLAS-compiled GEMM.

**Driver.**

```bash
bash scripts/ae/run_e4.sh
python scripts/ae/plot_fig12.py --root results/ae/e4 --out fig12.pdf
```

**Single-cell command.**

```bash
python -m benchmark.ae.run_e4 --batch-size <bs> --backend <tgx-pipe|tgx-no-pipe|cublas>
```

`tgx-no-pipe` requires a build with `-DMPK_PIPELINED_WORKER=OFF`. The driver
script handles the rebuild dance; a second pre-built install path is also
supported via `MPK_NO_PIPE_INSTALL=/path/to/legacy/install`.

---

## E5 — Ablation: compute–communication overlap (Fig. 13, §6.6)

**Claim.** Capturing fine-grained dependencies between collective and compute
tasks (and overlapping them) reduces per-iteration latency on Qwen3-1.7B
(4× H100, TP=4) by **1.1×** across batch sizes.

**Driver.**

```bash
bash scripts/ae/run_e5.sh
python scripts/ae/plot_fig13.py --root results/ae/e5 --out fig13.pdf
```

**Single-cell command.**

```bash
# With overlap (default)
mpirun -n 4 --allow-run-as-root \
    python demo/qwen3/demo.py --use-mirage \
        --model Qwen/Qwen3-1.7B --model-path /tmp/qwen3-1.7b \
        --max-num-batched-requests <bs> \
        --max-new-tokens 1024 --temperature 0

# Without overlap
mpirun -n 4 --allow-run-as-root \
    python demo/qwen3/demo.py --use-mirage --no-overlap \
        --model Qwen/Qwen3-1.7B --model-path /tmp/qwen3-1.7b \
        --max-num-batched-requests <bs> \
        --max-new-tokens 1024 --temperature 0
```

---

## Reproduction notes

- **First TGX run is slow.** It triggers compilation + Mirage
  superoptimization; later runs reuse the cached megakernel under
  `~/.cache/mirage/`. To pre-warm, run with `--max-new-tokens 4` once.
- **Run-to-run variance.** Sweep drivers run 5 reps and report the median.
- **HuggingFace gating.** Llama-3.2 needs an HF token (`HF_TOKEN=<token>`).
- **GPU access.** Modal offers A100 (40 GB / 80 GB), H100 (1/2/4/8), and
  B200 — sufficient to reproduce every cell of the evaluation matrix.
- **Qwen3-30B-A3B + A100** is omitted in Fig. 9 due to OOM.

## Layout

```
scripts/ae/
├── run_e1.sh ... run_e5.sh           # per-experiment sweep drivers
├── plot_fig9.py ... plot_fig13.py    # plotting (consume results/ae/*.json)
└── baselines/
    ├── run_vllm.sh
    └── run_sglang.sh
benchmark/ae/run_e4.py                # microbench driver for Fig. 12
results/ae/                           # JSON outputs (gitignored)
```
