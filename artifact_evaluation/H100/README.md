# H100 — single-GPU sweeps (E1 row 2 of Fig. 9)

Reproduces the H100 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, models Qwen3-{0.6B, 1.7B, 8B,
30B-A3B} + Llama-3.2-1B-Instruct).

## Demo entry points

H100 uses the **Hopper-tuned** demo files (WGMMA-aware grid sizing,
lm_head split):

| Model                 | Demo file                              |
|-----------------------|----------------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo_hopper.py`            |
| Qwen3-30B-A3B (MoE)   | `demo/qwen3/demo_30B_A3B_hopper.py`    |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`                  |

The Hopper demos do not support `--save-tokens`, so latency is parsed
from the `per-token latency ... ms` stdout line. Decode length is set
via `--max-seq-length 1088` (= prompt 64 + generate 1024).

### Known limitation: Qwen3-30B-A3B at batch sizes >1

The Hopper MoE megakernel for Qwen3-30B-A3B is shape-tuned for
`max_num_batched_tokens=1` (one token per request per decode iter). At
larger batch sizes the kernel processes requests serially within an
iter, so 30B-A3B's per-token latency scales linearly with batch size
(bs=N is roughly N× bs=1) instead of the sub-linear scaling other
models show. Forcing larger `max_num_batched_tokens` causes the kernel
to silently no-op (output stays as the eos sentinel). Fix would require
C++ kernel changes; out of scope for this artifact.

## How to run (any H100 host with CUDA 12.4)

These instructions work on **any** Linux + H100 host: bare metal,
Lambda, Crusoe, on-prem cluster. They do not depend on Modal.

### Prerequisites

- H100 80 GB with NVIDIA driver supporting CUDA 12.4
- Ubuntu 22.04+ with Python 3.10+
- HuggingFace token (`HF_TOKEN`) for Llama-3.2 (gated)

### One-shot setup

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx
```

### TGX/MPK + PyTorch sweeps (~75 min total)

```bash
bash artifact_evaluation/H100/run_tgx.sh
bash artifact_evaluation/H100/run_pytorch.sh
```

### vLLM + SGLang baselines

In a separate Python venv (vLLM/SGLang conflict with flashinfer's torch
pin):

```bash
python3 -m venv /opt/baselines-venv
source /opt/baselines-venv/bin/activate
pip install --upgrade pip
pip install vllm
pip install 'sglang[all]'

bash artifact_evaluation/H100/run_vllm.sh
bash artifact_evaluation/H100/run_sglang.sh
deactivate
```

### Filtering

```bash
MODELS=qwen3-0.6b BATCH_SIZES=1 bash artifact_evaluation/H100/run_tgx.sh
```

### Output schema

Same as A100 (`results/H100/<system>/<model_tag>__bs<bs>.json` with
`latency_ms_per_token`).

## Coverage on H100

| Experiment | Covered? | Notes |
|-----------|----------|-------|
| E1 (Fig. 9 H100 row) | ✅ | 5 models × 5 batch sizes × 4 systems |
| E3 (Fig. 11)         | ❌ | Multi-GPU (2/4/8 × H100); separate folder |
| E5 (Fig. 13)         | ❌ | 4× H100 ablation; separate folder |
| E2, E4               | ❌ | B200-only |

---

## Optional: launching on Modal cloud GPUs

This repo includes Modal helpers if you don't want to provision your
own host. Entirely optional.

```bash
# In a local terminal (your laptop)
modal run scripts/ae/ae_ssh.py --gpu h100
# prints:  SSH ready:  ssh root@<host>.modal.host -p <port>

# Paste the printed ssh line in another terminal, then run the same
# setup.sh + sweep commands as above. Results land at /mirage/results
# which is backed by the `tgx-ae-results` Modal Volume.
```
