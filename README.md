# Artifact Evaluation — TGX (OSDI '26 paper #804)

*TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*

> **Branch `tgx-osdi26-ae`** — the frozen artifact for OSDI '26 AE.
> For the general Mirage project README, see the
> [`mpk` branch](https://github.com/mirage-project/mirage/tree/mpk).

This README documents (1) what's reproduced, (2) how to run on any
GPU host, and (3) optional shortcuts for running on Modal cloud GPUs.

---

## What's reproduced

### Models, batch sizes, GPUs

| Paper name             | HuggingFace ID                       | Demo entry point                        |
|------------------------|--------------------------------------|-----------------------------------------|
| Qwen3-0.6B             | `Qwen/Qwen3-0.6B`                    | `demo/qwen3/demo.py` (A100) / `demo_hopper.py` (H100) |
| Llama-3.2-1B-Instruct  | `meta-llama/Llama-3.2-1B-Instruct`   | `demo/llama3/demo.py`                   |
| Qwen3-1.7B             | `Qwen/Qwen3-1.7B`                    | `demo/qwen3/demo.py` / `demo_hopper.py` |
| Qwen3-8B               | `Qwen/Qwen3-8B`                      | `demo/qwen3/demo.py` / `demo_hopper.py` |
| Qwen3-30B-A3B (MoE)    | `Qwen/Qwen3-30B-A3B`                 | `demo/qwen3/demo_30B_A3B.py` / `demo_30B_A3B_hopper.py` |

**Batch sizes:** 1, 2, 4, 8, 16 (per the paper's offline batched setup).

**GPU variants:** NVIDIA A100-40GB, NVIDIA H100-80GB SXM, NVIDIA B200.

**Systems compared:** TGX, PyTorch, vLLM, SGLang.

**Workload.** Offline batched inference, prompt length **64**, decode
**1024** tokens, batch sizes **{1, 2, 4, 8, 16}**, greedy
(`--temperature 0`). Numbers in the paper are the median of 5 runs after
a 4-iteration warmup.

**Metric.** Per-token decoding latency in ms, parsed from each demo's
stdout line:

```
Prompt length 64, generate length 1024, per-token latency (both prefill and decode): X.XXX ms
```

All baselines are converted to the same metric. Each cell produces a
JSON of the shape `{system, gpu, model, batch_size, latency_ms_per_token, ...}`.

**TGX configuration (Table 1, set automatically per GPU).**

| GPU  | # SMs | # workers | # schedulers | Shared-mem page |
|------|-------|-----------|--------------|------|
| A100 | 108   | 104       | 16           | 32 KB |
| H100 | 132   | 128       | 16           | 32 KB |
| B200 | 148   | 144       | 16           | 32 KB |

---

## How to run on any GPU host

Per-GPU instructions live in:

- [`artifact_evaluation/A100/README.md`](artifact_evaluation/A100/README.md)
- [`artifact_evaluation/H100/README.md`](artifact_evaluation/H100/README.md)
- [`artifact_evaluation/B200/README.md`](artifact_evaluation/B200/README.md)

Each folder is self-contained and assumes a Linux + CUDA host already
provisioned. The minimum flow:

```bash
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx                                    # for Llama-3.2 (gated)
bash artifact_evaluation/<gpu>/run_tgx.sh                  # TGX
bash artifact_evaluation/<gpu>/run_pytorch.sh              # PyTorch baseline
```

For vLLM and SGLang, install each in its own venv (they have
incompatible torch / transformers pins) — see the per-GPU README.

`setup.sh` clones the branch into `/mirage`, installs apt + pip deps,
auto-detects the GPU compute capability, and builds TGX. Takes
~10–15 min on first run.

---

## Optional: cloud hosting on Modal

If you don't have a GPU host, this artifact includes helpers to spin
up Modal cloud GPUs. **This is entirely optional**; the per-GPU
instructions above work on any A100/H100/B200 host.

If you don't have a Modal account, contact the authors and we can
provide temporary access to a pre-configured Modal instance.

### Local prereqs (one-time)

```bash
pip install modal
modal setup    # browser-based auth
```

Make sure `~/.ssh/id_rsa.pub` exists (`ssh-keygen -t rsa` if not).

### Start a GPU box

```bash
# pick one
modal run scripts/ae/ae_ssh.py --gpu a100-40gb
modal run scripts/ae/ae_ssh.py --gpu a100-80gb
modal run scripts/ae/ae_ssh.py --gpu h100
modal run scripts/ae/ae_ssh.py --gpu h100x4
modal run scripts/ae/ae_ssh.py --gpu h100x8
modal run scripts/ae/ae_ssh.py --gpu b200
```

The launcher prints:

```
SSH ready:  ssh root@<host>.modal.host -p <port>
```

In another terminal, paste that ssh line; you're now on a fresh
Modal-hosted GPU box. Run the per-GPU `setup.sh` + sweep scripts as in
the previous section.

### Persistent volumes

The Modal container mounts:

- `/root/.cache/huggingface` → `tgx-ae-hf-cache` Volume (HF weights
  reused across runs)
- `/mirage/results` → `tgx-ae-results` Volume (all benchmark JSONs
  persist across container restarts)

The launcher commits the results volume every 2 minutes to survive
ungraceful container exits (OOM, force-stop, etc.).

## Layout

```
artifact_evaluation/
├── setup.sh              # bootstrap TGX on a fresh GPU host
├── A100/                 # Fig. 9 row 3 (4 models × 5 bs × 4 systems)
├── H100/                 # Fig. 9 row 2 (5 models × 5 bs × 4 systems)
└── B200/                 # Fig. 9 row 1 (5 models × 5 bs × 4 systems)
scripts/ae/
├── ae_ssh.py             # Optional: Modal SSH launcher (cloud helper)
└── ae_modal.py           # Optional: Modal one-shot launcher
```
