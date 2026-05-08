# H100 — single-GPU sweeps (E1 row 2 of Fig. 9)

Reproduces the H100 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, models Qwen3-{0.6B, 1.7B, 8B,
30B-A3B} + Llama-3.2-1B-Instruct).

## Demo entry points

The `run_tgx.sh` and `run_pytorch.sh` scripts dispatch to **Hopper-tuned**
demo files (WGMMA-aware grid sizing, lm_head split):

| Model                 | Demo file                              |
|-----------------------|----------------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo_hopper.py`            |
| Qwen3-30B-A3B (MoE)   | `demo/qwen3/demo_30B_A3B_hopper.py`    |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`                  |

The Hopper demos do not support `--save-tokens`, so latency is parsed
from the demo's stdout line `per-token latency ... X.XXX ms`. Decode
length is set via `--max-seq-length 1088` (= prompt 64 + generate 1024).

Each script writes one JSON per (model, batch_size) cell to
`results/H100/<system>/<model_tag>__bs<bs>.json`. JSON schema:

```json
{
  "system":   "tgx" | "pytorch" | "vllm" | "sglang",
  "model":    "<HuggingFace ID>",
  "batch_size": <int>,
  "latency_ms_per_token": <float>,
  ...
}
```

## Running on Modal

TGX + PyTorch share the TGX Modal image (`run_h100`):

```bash
modal run scripts/ae/ae_modal.py::run_h100 \
    --cmd "bash artifact_evaluation/H100/run_tgx.sh"

modal run scripts/ae/ae_modal.py::run_h100 \
    --cmd "bash artifact_evaluation/H100/run_pytorch.sh"
```

vLLM + SGLang use the baseline Modal image (`baseline_h100`) because
their dependency cone conflicts with flashinfer:

```bash
modal run scripts/ae/ae_modal.py::baseline_h100 \
    --cmd "bash artifact_evaluation/H100/run_vllm.sh"

modal run scripts/ae/ae_modal.py::baseline_h100 \
    --cmd "bash artifact_evaluation/H100/run_sglang.sh"
```

## Filtering

Both per-system scripts respect `MODELS` and `BATCH_SIZES` env vars.
Useful for spot checks before committing to the full ~10 GPU-hour run:

```bash
modal run scripts/ae/ae_modal.py::run_h100 \
    --cmd "MODELS=qwen3-0.6b BATCH_SIZES=1 bash artifact_evaluation/H100/run_tgx.sh"
```

## Persisted results

Output files land under `/mirage/results` inside the container, which is
mounted to the `tgx-ae-results` Modal Volume — so JSONs survive across
container restarts and can be downloaded with `modal volume get`.

## Coverage on H100

| Experiment | Covered by these scripts? | Notes |
|-----------|---------------------------|-------|
| E1 (Fig. 9 H100 row) | ✅ | All 4 systems × 5 models × 5 batch sizes |
| E3 (Fig. 11)         | ❌ | Multi-GPU; needs `run_h100x{2,4,8}` and a separate sweep script |
| E5 (Fig. 13)         | ❌ | 4× H100 with overlap on/off; separate script needed |
| E2, E4               | ❌ | B200-only |
