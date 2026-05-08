# Artifact Evaluation — TGX (OSDI '26 paper #804)

*TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*

This file is the AE entry point. It documents (1) what's reproduced,
(2) how to run on any GPU host, and (3) optional shortcuts for running
on Modal cloud GPUs.

> **Branch.** `tgx-osdi26-ae` — the frozen artifact for OSDI '26 AE.

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

**Systems compared:** TGX (= MPK, our system), PyTorch (the same demos
run without `--use-mirage`), vLLM (`vllm bench latency`), SGLang
(`python -m sglang.bench_one_batch`). All four systems write the same
JSON schema with `latency_ms_per_token`.

### Experiments

| Exp. | Paper figure  | What it covers | Folder                          |
|------|---------------|----------------|---------------------------------|
| E1   | Fig. 9 (§6.3) | 5 models × 5 batch sizes × 4 systems on each of A100/H100/B200 (Qwen3-30B-A3B omitted on A100 per paper). 70 cells total. | `artifact_evaluation/{A100,H100,B200}/` |
| E2   | Fig. 10 (§6.4)| Qwen3-30B-A3B MoE on B200, 5 batch sizes × 3 configurations (SGLang-MoE, TGX-Static, TGX-Hybrid). | `artifact_evaluation/B200/`     |
| E3   | Fig. 11 (§6.5)| Qwen3-1.7B with TP on 2 / 4 / 8 × H100, 5 batch sizes × 4 systems. | `artifact_evaluation/H100xN/`   |
| E4   | Fig. 12 (§6.6)| Qwen3-8B final linear layer on B200, 5 batch sizes × 3 configs (cuBLAS, TGX-No-Pipe, TGX-Pipe). | `artifact_evaluation/B200/`     |
| E5   | Fig. 13 (§6.6)| Qwen3-1.7B on 4 × H100 with TP, 5 batch sizes × 2 configs (TGX with/without compute–comm overlap). | `artifact_evaluation/H100xN/`   |

### Reproduction status (current snapshot)

| Sweep | Cells | Status |
|-------|-------|--------|
| A100 TGX (4 models × 5 bs)            | 20 | ✅ done |
| A100 PyTorch (4 models × 5 bs)        | 20 | ✅ done |
| A100 vLLM (4 models × 5 bs)           | 20 | 🔄 in progress |
| A100 SGLang (4 models × 5 bs)         | 20 | 🔄 queued behind vLLM |
| H100 TGX (5 models × 5 bs)            | 25 | ⏳ pending re-run |
| H100 PyTorch (5 models × 5 bs)        | 25 | 🔄 in progress |
| H100 vLLM (5 models × 5 bs)           | 25 | 🔄 queued |
| H100 SGLang (5 models × 5 bs)         | 25 | 🔄 queued |
| B200 (E1 row 1, E2, E4)                | tbd| ⏳ not started |
| H100 multi-GPU (E3, E5)                | tbd| ⏳ not started |

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
- *(B200 / multi-GPU folders coming as we complete those experiments)*

Each folder is self-contained and assumes a Linux + CUDA 12.4 host
already provisioned. The minimum flow is:

```bash
# inside the GPU host (any A100/H100/B200, bare metal or cloud)
curl -sSL https://raw.githubusercontent.com/mirage-project/mirage/tgx-osdi26-ae/artifact_evaluation/setup.sh | bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_TOKEN=hf_xxx                                    # for Llama-3.2 (gated)
bash artifact_evaluation/<gpu>/run_tgx.sh                  # TGX/MPK
bash artifact_evaluation/<gpu>/run_pytorch.sh              # PyTorch baseline

# vLLM and SGLang share a separate Python venv (their torch pin
# conflicts with flashinfer's)
python3 -m venv /opt/baselines-venv
source /opt/baselines-venv/bin/activate
pip install --upgrade pip vllm 'sglang[all]'
bash artifact_evaluation/<gpu>/run_vllm.sh
bash artifact_evaluation/<gpu>/run_sglang.sh
```

`setup.sh` clones the branch into `/mirage`, installs apt + pip deps,
and builds MPK. Takes ~10–15 min on first run.

---

## Optional: cloud hosting on Modal

If you don't have a GPU host, this artifact includes helpers to spin
up Modal cloud GPUs. **This is entirely optional**; the per-GPU
instructions above work on any A100/H100/B200 host.

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

### Caveats

- Image build is ~10 min the first time per cache.
- A bare GPU host avoids Modal-specific concerns (volumes, auth,
  per-second billing). Use whichever fits your workflow.

---

## Reproduction notes

- **First TGX run is slow.** Triggers a one-time NVCC compile of the
  megakernel (~1–5 min depending on model). Cached under
  `~/.cache/mirage/`. To pre-warm, run with a small `--max-seq-length`
  once.
- **Run-to-run variance.** Sweep drivers run a built-in warmup before
  timing.
- **HuggingFace gating.** Llama-3.2 requires accepting Meta's license.
  Set `HF_TOKEN=<your_token>` before the sweep.
- **Qwen3-30B-A3B on A100** is omitted in Fig. 9 (paper §6.2) due to
  OOM on a 40 GB A100.
- **Qwen3-30B-A3B on H100** at batch sizes >1 currently scales linearly
  with batch (kernel limitation, see `artifact_evaluation/H100/README.md`).

---

## Layout

```
artifact_evaluation/
├── setup.sh              # bootstrap MPK on a fresh GPU host
├── A100/                 # E1 row 3 of Fig. 9 (4 models × 5 bs × 4 systems)
├── H100/                 # E1 row 2 of Fig. 9 (5 models × 5 bs × 4 systems)
├── ...                   # B200, H100xN added as experiments complete
scripts/ae/
├── ae_ssh.py             # Optional: Modal SSH launcher (cloud helper)
└── ae_modal.py           # Optional: Modal one-shot launcher
```
