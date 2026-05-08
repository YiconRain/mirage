# A100 — single-GPU sweeps (E1 row 3 of Fig. 9)

Reproduces the A100 row of Fig. 9 (per-token decode latency, prompt = 64,
generate = 1024, batch sizes 1/2/4/8/16, models Qwen3-{0.6B, 1.7B, 8B} +
Llama-3.2-1B-Instruct). **Qwen3-30B-A3B is omitted by default** — the paper
notes it OOMs on a single A100.

## Demo entry points

A100 uses the Ampere demo files (also what CI runs):

| Model                 | Demo file                       |
|-----------------------|---------------------------------|
| Qwen3-0.6B / 1.7B / 8B| `demo/qwen3/demo.py`            |
| Qwen3-30B-A3B (off)   | `demo/qwen3/demo_30B_A3B.py`    |
| Llama-3.2-1B-Instruct | `demo/llama3/demo.py`           |

## How to run

Use the standard 40GB A100 (paper config; matches "model exceeds memory
capacity of a single A100" omission for 30B-A3B).

```bash
modal run scripts/ae/ae_ssh.py --gpu a100-40gb
# prints:  SSH ready:  ssh root@<host>.modal.host -p <port>

ssh root@<host>.modal.host -p <port>
bash <(curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx     # for Llama-3.2

bash artifact_evaluation/A100/run_tgx.sh
bash artifact_evaluation/A100/run_pytorch.sh
```

For the baseline sweeps (vLLM / SGLang) use the one-shot launcher:

```bash
modal run scripts/ae/ae_modal.py::baseline_a100_40gb \
    --cmd "bash artifact_evaluation/A100/run_vllm.sh"
modal run scripts/ae/ae_modal.py::baseline_a100_40gb \
    --cmd "bash artifact_evaluation/A100/run_sglang.sh"
```

## Filtering

Override `MODELS` or `BATCH_SIZES`:

```bash
MODELS=qwen3-0.6b BATCH_SIZES=1 bash artifact_evaluation/A100/run_tgx.sh
```

## Coverage on A100

| Experiment | Covered? | Notes |
|-----------|----------|-------|
| E1 (Fig. 9 A100 row) | ✅ | 4 models × 5 batch sizes × 4 systems (no 30B-A3B) |
| All others           | ❌ | E2/E3/E5 are H100; E4 is B200 |
