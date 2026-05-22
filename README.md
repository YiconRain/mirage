# Artifact Evaluation — TGX (OSDI '26 paper #804)

*TGX: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*

> **Branch `tgx-osdi26-ae`** — the frozen artifact for OSDI '26 AE.

This README documents (1) what's reproduced and (2) how to run on any
GPU host.

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

## Multi-GPU prerequisites — NVSHMEM + MPI

The multi-GPU experiments (**Fig. 11** cross-system TP comparison and
**Fig. 13** compute–communication overlap ablation) require NVSHMEM
and an MPI implementation. The standalone `pip install -e .` workflow
used by Fig. 9 does **not** pull these in — install them once before
running the multi-GPU sweeps.

### What MPK looks up

At compile time `mpk.PersistentKernel.compile()` reads four env vars and
validates the corresponding files exist (see
[`python/mirage/mpk/persistent_kernel.py:2376-2442`](python/mirage/mpk/persistent_kernel.py#L2376-L2442)):

| Env var             | Validates                                      | Falls back to                  |
|---------------------|------------------------------------------------|--------------------------------|
| `NVSHMEM_INC_PATH`  | `$NVSHMEM_INC_PATH/nvshmem.h`                  | `/usr/include/nvshmem_12/`     |
| `NVSHMEM_LIB_PATH`  | `$NVSHMEM_LIB_PATH/libnvshmem_device.a`        | `/usr/lib/x86_64-linux-gnu/`   |
| `MPI_INC_PATH`      | `$MPI_INC_PATH/mpi.h`                          | `/usr/include/`                |
| `MPI_LIB_PATH`      | `$MPI_LIB_PATH/libmpi.so`                      | `/usr/lib/`                    |

The generated `nvcc` command links with `-ccbin=mpic++ -lnvshmem_host
-lnvshmem_device -lmpi` and bakes `$NVSHMEM_LIB_PATH` and `$MPI_LIB_PATH`
into `DT_RPATH` (with `--disable-new-dtags`).

### Install NVSHMEM

The compiler needs **both** the host library (`libnvshmem_host.so`) and
the device library (`libnvshmem_device.a`). The redistributable archive
from NVIDIA Developer (`libnvshmem-linux-x86_64-<ver>_cuda<cuda>-archive`)
contains both. Pick the CUDA major matching your toolkit (e.g.
`cuda13` for CUDA 12.x/13.x, `cuda12` for CUDA 12.x).

```bash
# 1. Download the prebuilt archive (no sudo needed).
NVSHMEM_VER=3.6.5
NVSHMEM_CUDA=cuda13
mkdir -p $HOME/lib && cd $HOME/lib
wget -q https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VER}/local_installers/libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive.tar.xz
tar xJf libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive.tar.xz
export NVSHMEM_HOME=$HOME/lib/libnvshmem-linux-x86_64-${NVSHMEM_VER}_${NVSHMEM_CUDA}-archive

# 2. Export the env vars MPK reads.
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib
export NVSHMEM_PREFIX=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}

# 3. Verify.
ls $NVSHMEM_INC_PATH/nvshmem.h $NVSHMEM_LIB_PATH/libnvshmem_device.a
```

### Install MPI

Any MPI ≥ 3.1 works. The simplest path is conda-installed Open MPI
(no root needed); on hosts where MPI is already system-installed
(`apt install libopenmpi-dev`), point the env vars at the system paths
instead.

```bash
# Option A — conda (recommended for non-root hosts; assumes the
# project conda env, e.g. mirage_2, is active):
conda install -y -c conda-forge openmpi mpi4py
export MPI_HOME=$CONDA_PREFIX
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib

# Option B — Ubuntu system Open MPI:
sudo apt install -y libopenmpi-dev openmpi-bin
export MPI_HOME=/usr
export MPI_INC_PATH=/usr/include/openmpi
export MPI_LIB_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib

# Verify mpic++ is on PATH and resolves to the same install.
which mpic++ mpirun
ls $MPI_INC_PATH/mpi.h $MPI_LIB_PATH/libmpi.so
```

The Python multi-GPU launch path uses `mpi4py`; the conda command
above pulls it in. With Option B, also run `pip install mpi4py`.

### Persist the env vars

Add the exports to `~/.bashrc` (or a project setup script) so every
shell that runs the AE scripts inherits them:

```bash
cat >> ~/.bashrc <<'EOF'
# MPK multi-GPU build deps
export NVSHMEM_HOME=$HOME/lib/libnvshmem-linux-x86_64-3.6.5_cuda13-archive
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib
export NVSHMEM_PREFIX=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export MPI_HOME=$CONDA_PREFIX                  # adjust if not using conda
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib
EOF
```

Once `NVSHMEM_LIB_PATH` is set, `run_fig13_overlap.sh` /
`run_fig11_multigpu.sh` automatically prepend the matching
`libnvshmem_host.so.3` to `LD_PRELOAD` — this avoids picking up an
older `/usr/lib` NVSHMEM at runtime (the well-known
`undefined symbol: nvshmem_selected_device_transport` import error).

### Rebuild MPK if any of these change

You do not need to rebuild the C++ runtime library after installing or
upgrading NVSHMEM / MPI — each kernel cell re-emits its `nvcc`
command with the currently-set paths. Only edits under `src/` require
`pip install -e . --no-deps -v`.

---

## Optional: cloud hosting on Modal

If you don't have a GPU host, contact the authors — we can provide
Modal access.

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

---

## Additional experiments (Fig. 10 + Fig. 12)

The default sweeps above reproduce **Fig. 9** (per-token decode latency
across 5 models × 5 batch sizes × 3 GPUs × 4 systems). Two additional
experiments from the paper have their own per-GPU scripts in the same
folders, both currently provided for **B200**:

### Fig. 10 — MoE microbenchmark (Qwen3-30B-A3B)

Compares MPK's hybrid workload balancer + fused gather-GEMM to SGLang's
MoE implementation, on a single B200.

```bash
bash artifact_evaluation/B200/run_fig10_moe.sh        # MPK-Hybrid-MoE (TGX)
bash artifact_evaluation/B200/run_sglang.sh \
    MODELS=qwen3-30b-a3b OUTPUT_ROOT=results/B200/fig10/sglang   # SGLang-MoE
```

Outputs land in `results/B200/fig10/<system>__bs<bs>.json` with
`latency_ms_per_token`. Qwen3-30B-A3B is MoE-dominated, so per-token
latency tracks MoE-block runtime closely.

The paper's third bar, `MPK-Static-MoE`, is an internal ablation of the
hybrid workload balancer and is not exposed as a separate runtime flag
in this artifact. For the per-MoE-block μs numbers in the paper, profile
the `moe_w13_linear` / `moe_w2_linear` kernels with NCU:

```bash
ncu --kernel-name regex:'moe_w[12]3?_linear' --launch-count 4 \
    python demo/qwen3/demo_30B_A3B.py --use-mirage \
    --model Qwen/Qwen3-30B-A3B --max-num-batched-requests 1 \
    --max-seq-length 128 --ignore-eos
```

**Note on the historical MAX_TOKENS quirk.** Earlier drafts of this
demo required a manual edit of
`include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh`
to set `MAX_TOKENS = 1` for Qwen3-30B-A3B's high GQA group ratio. This
is now handled automatically at compile time in `src/kernel/task_register.cc`
(see commit 688632e): when `NUM_QO_PER_KV >= 8`, the task registry
instantiates the attention kernel template with `MAX_TOKENS=4`. No
manual rebuild needed.

### Fig. 12 — Cross-task pipelining ablation (Qwen3-8B lm_head)

Compares two MPK configurations differing only in the grid size of the
final `lm_head` linear layer:

- `MPK-Pipe` (`grid_dim[0] = 128`): few enough tasks that the cross-task
  pipeliner engages — the next layer's pre-load overlaps the current
  layer's compute.
- `MPK-No-Pipe` (`grid_dim[0] = vocab_size // 256 = 600`): too many tasks
  for pipelining to fit.

Run both with one script:

```bash
bash artifact_evaluation/B200/run_fig12_pipe_ablation.sh
```

Outputs `results/B200/fig12/mpk-pipe__bs<bs>.json` and
`results/B200/fig12/mpk-no-pipe__bs<bs>.json`. The `MPK-Pipe` minus
`MPK-No-Pipe` per-token-latency delta approximates the lm-head
pipelining benefit. For the absolute per-layer μs numbers in the
paper, profile the final `linear_layer` kernel with NCU:

```bash
ncu --kernel-name regex:'.*linear.*' --launch-count 8 \
    python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-8B \
    --max-num-batched-requests 1 --max-seq-length 128 \
    --lm-head-grid pipe --ignore-eos
# repeat with --lm-head-grid no-pipe
```

The `--lm-head-grid` knob is added to `demo/qwen3/demo.py`. It defaults
to `default` (`mpk.num_workers`), so existing Fig. 9 sweeps are
unaffected.

### CUBLAS baseline (third bar in Fig. 12)

The PyTorch + cuBLAS curve in Fig. 12 is the same Qwen3-8B Fig. 9 cell:

```bash
bash artifact_evaluation/B200/run_pytorch.sh MODELS=qwen3-8b
# JSONs land in results/B200/pytorch/qwen3-8b__bs<bs>.json
```

The `latency_ms_per_token` value for each batch size is the
`CUBLAS` curve.

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
├── setup.sh              # bootstrap TGX on a fresh GPU host
├── A100/                 # 4 models × 5 batch sizes × 4 systems
├── H100/                 # 5 models × 5 batch sizes × 4 systems
└── B200/                 # 5 models × 5 batch sizes × 4 systems
scripts/ae/
├── ae_ssh.py             # Optional: Modal SSH launcher (cloud helper)
└── ae_modal.py           # Optional: Modal one-shot launcher
```
