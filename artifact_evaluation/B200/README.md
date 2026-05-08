# B200 — single-GPU sweeps (E1 row 1 of Fig. 9)

Reproduces the B200 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, 5 models).

## Demo entry points

| Model                 | Demo file                        |
|-----------------------|----------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo_blackwell.py`   |
| Qwen3-30B-A3B (MoE)   | `demo/qwen3/demo_30B_A3B.py`     |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`            |

The Blackwell demo (`demo_blackwell.py`) targets compute capability 10.0
(SM_100) with appropriate grid sizing. Decode length is set via
`--max-seq-length 1088` (= prompt 64 + generate 1024).

**Note on Qwen3-30B-A3B:** there is no Blackwell-tuned MoE demo yet, so
the sweep falls back to `demo/qwen3/demo_30B_A3B.py` (Ampere variant).
This may be suboptimal but produces real numbers; replace with a
Blackwell MoE demo when available.

## How to run (any B200 host with CUDA 12.4)

### Prerequisites

- NVIDIA B200 with driver supporting CUDA 12.4
- Ubuntu 22.04+ with Python 3.10+
- HuggingFace token (`HF_TOKEN`) for Llama-3.2 (gated)

### One-shot setup

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export HF_TOKEN=hf_xxx
```

### TGX/MPK + PyTorch sweeps

```bash
bash artifact_evaluation/B200/run_tgx.sh
bash artifact_evaluation/B200/run_pytorch.sh
```

### vLLM + SGLang baselines

vLLM/SGLang ship their own torch wheels; install them in a separate
Python venv. B200 needs CUDA 12.8 + torch 2.7 wheels (sm_100 support);
vLLM 0.10+ provides this.

```bash
python3 -m venv /opt/baselines-venv
source /opt/baselines-venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
pip install vllm
pip install 'sglang[all]'
deactivate

bash artifact_evaluation/B200/run_vllm.sh
bash artifact_evaluation/B200/run_sglang.sh
```

The `run_vllm.sh` and `run_sglang.sh` scripts auto-activate
`/opt/baselines-venv` when it exists.

### Filtering

```bash
MODELS=qwen3-0.6b BATCH_SIZES=1 bash artifact_evaluation/B200/run_tgx.sh
```

### Output schema

Same as A100/H100 (`results/B200/<system>/<model_tag>__bs<bs>.json`
with `latency_ms_per_token`).

If you don't have a B200 host, see the **Optional: cloud hosting on
Modal** section in the top-level [`AE_README.md`](../../AE_README.md).
