# B200 — single-GPU sweeps (E1 row 1 of Fig. 9)

Reproduces the B200 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, 5 models).

## Demo entry points

| Model                 | Demo file                        |
|-----------------------|----------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo.py`             |
| Qwen3-30B-A3B (MoE)   | `demo/qwen3/demo_30B_A3B.py`     |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`            |

The TGX runtime auto-selects the Blackwell task headers (sm_100) at
NVCC compile time via `-DMIRAGE_GRACE_BLACKWELL`. Decode length is set
via `--max-seq-length 1088` (= prompt 64 + generate 1024).

## How to run (any B200 host with CUDA 12.8+)

### Prerequisites

- NVIDIA B200 with driver supporting CUDA 12.8
- Ubuntu 22.04+ with Python 3.10+
- HuggingFace token (`HF_TOKEN`) for Llama-3.2 (gated)

### One-shot setup

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx
```

`setup.sh` auto-detects the GPU compute capability from `nvidia-smi`.
On B200 (sm_100) it installs torch 2.7.0+cu128 + flashinfer for
torch2.7/cu128. On Ampere/Hopper it installs torch 2.6.0+cu124. No
manual torch upgrade is needed.

### TGX + PyTorch sweeps

```bash
bash artifact_evaluation/B200/run_tgx.sh
bash artifact_evaluation/B200/run_pytorch.sh
```

### vLLM + SGLang baselines

vLLM and SGLang have fundamentally incompatible dependency cones
(different torch / transformers versions). Install each in its **own**
venv — `run_vllm.sh` auto-activates `/opt/vllm-venv` and
`run_sglang.sh` auto-activates `/opt/sglang-venv`.

**vLLM venv** (B200 needs cu128 + sm_100; vLLM 0.10+ ships this):

```bash
rm -rf /opt/vllm-venv
python3 -m venv /opt/vllm-venv
source /opt/vllm-venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
pip install vllm
deactivate

bash artifact_evaluation/B200/run_vllm.sh
```

**SGLang venv** (let pip resolve internally consistent set):

```bash
rm -rf /opt/sglang-venv
python3 -m venv /opt/sglang-venv
source /opt/sglang-venv/bin/activate
pip install --upgrade pip
pip install 'sglang[all]'
deactivate

bash artifact_evaluation/B200/run_sglang.sh
```

### Output schema

Same as A100/H100 (`results/B200/<system>/<model_tag>__bs<bs>.json`
with `latency_ms_per_token`).

If you don't have a B200 host, see the **Optional: cloud hosting on
Modal** section in the top-level [`README.md`](../../README.md).
