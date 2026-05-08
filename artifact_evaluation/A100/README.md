# A100 — single-GPU sweeps (E1 row 3 of Fig. 9)

Reproduces the A100 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, models Qwen3-{0.6B, 1.7B, 8B} +
Llama-3.2-1B-Instruct). **Qwen3-30B-A3B is omitted** — the paper notes it
OOMs on a single A100.

## Demo entry points

| Model                 | Demo file                       |
|-----------------------|---------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo.py`            |
| Qwen3-30B-A3B (off)   | `demo/qwen3/demo_30B_A3B.py`    |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`           |

## How to run (any A100 host with CUDA 12.4)

These instructions work on **any** Linux + A100 host: bare metal, Lambda,
Crusoe, GCP, on-prem cluster. They do not depend on Modal.

### Prerequisites

- A100 (40 GB or 80 GB) with NVIDIA driver supporting CUDA 12.4
- Ubuntu 22.04+ with Python 3.10+
- Network access (to clone GitHub + pull model weights from HuggingFace)
- HuggingFace token if you want Llama-3.2 (gated model)

### One-shot setup

ssh into the host, then:

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
```

`setup.sh` installs apt deps, clones mirage at branch `tgx-osdi26-ae`,
builds it, installs torch + transformers + flashinfer. Runtime ≈ 10–15 min
on first run (mostly CMake + NVCC compilation of the C++/CUDA extension).

After setup, in your shell:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx   # only needed for Llama-3.2 (gated)
```

### Run TGX/MPK + PyTorch sweeps

These two share the MPK environment (no extra deps):

```bash
bash artifact_evaluation/A100/run_tgx.sh        # ~30-40 min
bash artifact_evaluation/A100/run_pytorch.sh    # ~30-40 min
```

Each writes one JSON per (model, batch_size) cell to
`results/A100/<system>/<model_tag>__bs<bs>.json`.

### Run vLLM + SGLang baselines

vLLM/SGLang ship their own torch wheels and conflict with flashinfer,
so install them in a separate Python venv. Pin the versions below —
newer vLLM (≥0.9) ships torch built with CUDA 13 and breaks on CUDA
12.4 hosts.

```bash
python3 -m venv /opt/baselines-venv
source /opt/baselines-venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install vllm==0.8.5
pip install sglang==0.4.6.post5      # 0.5+ requires torch 2.11 / transformers 5.x
pip install 'transformers==4.51.3'   # vllm 0.8.5 needs transformers 4.x
deactivate

bash artifact_evaluation/A100/run_vllm.sh       # ~30-40 min
bash artifact_evaluation/A100/run_sglang.sh     # ~30-40 min
```

The `run_vllm.sh` and `run_sglang.sh` scripts auto-activate
`/opt/baselines-venv` when it exists, so no extra `source` is needed.

### Filtering / spot-checks

All sweep scripts respect `MODELS` and `BATCH_SIZES`:

```bash
MODELS=qwen3-0.6b BATCH_SIZES=1 bash artifact_evaluation/A100/run_tgx.sh
```

### Output schema

Each per-cell JSON looks like:

```json
{
  "system":   "tgx" | "pytorch" | "vllm" | "sglang",
  "gpu":      "A100",
  "model":    "<HuggingFace ID>",
  "batch_size": <int>,
  "prompt_len": 64,
  "gen_len":  1024,
  "latency_ms_per_token": <float>
}
```

## Coverage on A100

| Experiment | Covered? | Notes |
|-----------|----------|-------|
| E1 (Fig. 9 A100 row) | ✅ | 4 models × 5 batch sizes × 4 systems (no 30B-A3B) |
| Other experiments    | ❌ | E2/E4 are B200; E3/E5 are H100 |

If you don't have an A100 host, see the **Optional: cloud hosting on
Modal** section in the top-level [`AE_README.md`](../../AE_README.md).
