# A100 — single-GPU sweeps (E1 row 3 of Fig. 9)

Reproduces the A100 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, models Qwen3-{0.6B, 1.7B, 8B} +
Llama-3.2-1B-Instruct). **Qwen3-30B-A3B is omitted** — the paper notes it
OOMs on a single A100.

## Demo entry points

| Model                 | Demo file                       |
|-----------------------|---------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo.py`            |
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

**Non-root hosts** (shared clusters where `/mirage` and `/opt` aren't
writable): set `MIRAGE_HOME` and `SKIP_APT` first so setup clones into
your home dir and assumes apt deps are pre-installed:

```bash
export MIRAGE_HOME=$HOME/mirage-ae
export SKIP_APT=1
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
```

After setup, in your shell:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx   # only needed for Llama-3.2 (gated)
```

### Run TGX + PyTorch sweeps

These two share the TGX environment (no extra deps):

```bash
bash artifact_evaluation/A100/run_tgx.sh        # ~30-40 min
bash artifact_evaluation/A100/run_pytorch.sh    # ~30-40 min
```

Each writes one JSON per (model, batch_size) cell to
`results/A100/<system>/<model_tag>__bs<bs>.json`.

### Run vLLM + SGLang baselines

vLLM and SGLang have fundamentally incompatible dependency cones
(different torch / transformers versions). Install each in its **own
venv** — `run_vllm.sh` auto-activates `/opt/vllm-venv` and
`run_sglang.sh` auto-activates `/opt/sglang-venv`.

**vLLM venv** (pinned, since newer vLLM ships torch built for CUDA 13):

```bash
rm -rf /opt/vllm-venv
python3 -m venv /opt/vllm-venv
source /opt/vllm-venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install vllm==0.8.5
pip install transformers==4.51.3     # vllm 0.8.5 needs transformers 4.x
deactivate

bash artifact_evaluation/A100/run_vllm.sh       # ~30-40 min
```

**SGLang venv** (let pip resolve internally consistent set):

```bash
rm -rf /opt/sglang-venv
python3 -m venv /opt/sglang-venv
source /opt/sglang-venv/bin/activate
pip install --upgrade pip
pip install 'sglang[all]'
deactivate

bash artifact_evaluation/A100/run_sglang.sh     # ~30-40 min
```

**Non-root hosts** (e.g., shared clusters where `/opt` is not writable):
substitute `$HOME/vllm-venv` / `$HOME/sglang-venv` for `/opt/...` and
override the env vars:

```bash
BASELINES_VENV=$HOME/vllm-venv MIRAGE_HOME=$HOME/mirage-ae \
    bash $HOME/mirage-ae/artifact_evaluation/A100/run_vllm.sh
BASELINES_VENV=$HOME/sglang-venv MIRAGE_HOME=$HOME/mirage-ae \
    bash $HOME/mirage-ae/artifact_evaluation/A100/run_sglang.sh
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

If you don't have an A100 host, see the **Optional: cloud hosting on
Modal** section in the top-level [`README.md`](../../README.md).
