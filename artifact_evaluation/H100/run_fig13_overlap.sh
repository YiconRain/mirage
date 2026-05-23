#!/usr/bin/env bash
# Fig. 13 — compute–communication overlap ablation on H100.
#
# The paper plots per-iteration runtime of Qwen3-1.7B under tensor
# parallelism (4 H100s) with two MPK settings:
#   - MPK (with overlap)    : allgather-style allreduce; the allgather
#                             phase is allowed to overlap with downstream
#                             computation (default MPK behavior for the
#                             split-phase allreduce).
#   - MPK (without overlap) : same allreduce, but a single event is
#                             forced before and after the allgather task,
#                             serializing it with surrounding computation.
#
# Both modes use the AllgatherReduce strategy (NOT the NvshmemTile path
# that newer GPUs would otherwise auto-select), so the comparison isolates
# the overlap variable. Strategy and overlap are toggled by two env vars
# read by the MPK compiler:
#   MPK_FORCE_ALLGATHER_REDUCE=1  — Python (multigpu.py auto-select)
#   MPK_DISABLE_AG_OVERLAP=1      — C++   (annotated_graph.cc final pass)
#
# The script is the paper-matching artifact for H100 but is portable: on
# B200 hosts override GPU=B200 (and CUDA_VISIBLE_DEVICES accordingly) to
# reproduce the same trend.
#
# Output: results/H100/fig13/mpk-{overlap,no-overlap}__bs<bs>.json
# Each JSON carries latency_ms_per_token; plot_fig13.py converts to µs.
#
# Prerequisites:
#   - 4 idle GPUs (e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3).
#   - HF cache must contain Qwen/Qwen3-1.7B (override HF_HOME to point
#     at your cache).

set -euo pipefail

export PATH="${CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MIRAGE_HOME="${MIRAGE_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$MIRAGE_HOME"

GPU="${GPU:-H100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/$GPU/fig13}"
mkdir -p "$OUTPUT_ROOT"

# Use the shared HF cache on /raid by default so Qwen3-1.7B is found locally.
export HF_HOME="${HF_HOME:-/raid/catalyst/models}"

# Some systems ship an older libnvshmem in /usr/lib that is missing symbols
# MPK's compiled megakernel needs (e.g. nvshmem_selected_device_transport).
# When NVSHMEM_LIB_PATH is set (typical NVSHMEM install layout from ~/.bashrc),
# force-load the matching libnvshmem_host.so.3 so the dynamic linker doesn't
# pick up the older system copy.
if [[ -n "${NVSHMEM_LIB_PATH:-}" && -f "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" ]]; then
    export LD_PRELOAD="${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
fi

MODEL_TAG="${MODEL_TAG:-qwen3-1.7b}"
HF_ID="${HF_ID:-Qwen/Qwen3-1.7B}"
SCRIPT="${SCRIPT:-demo/qwen3/demo.py}"

WORLD_SIZE="${WORLD_SIZE:-4}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + GEN_LEN))}"

# Ablation modes. Both force AllgatherReduce; no-overlap additionally
# serializes the allgather edges via the C++ override pass.
MODES="${MODES:-overlap no-overlap}"

# mpirun env-propagation list. -x with no value forwards the current
# shell's value into each MPI rank.
MPIRUN_ENVS=(
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH
    -x HF_HOME
    -x MPK_FORCE_ALLGATHER_REDUCE -x MPK_DISABLE_AG_OVERLAP
    -x NCCL_NVLS_ENABLE       # Modal H100: NVLS transport OOMs at >30 nvshmem teams
)

parse_latency_ms() {
    grep -oiE 'per-token latency[^0-9]*[0-9]+\.[0-9]+ ms' "$1" \
        | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
}

run_cell() {
    local mode="$1"
    local bs="$2"
    local tag="mpk-${mode}"
    local log="$OUTPUT_ROOT/${tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${tag}__bs${bs}.json"

    # overlap mode = MPK's default auto-selected allreduce (NvshmemTile on
    # SM>=90 when VMM/multicast/peer-access are supported, with built-in
    # compute-comm overlap). no-overlap mode = force AllgatherReduce and
    # then collapse the allgather event edges so the allgather phase is
    # serialized with the rest of the megakernel.
    if [[ "$mode" == "no-overlap" ]]; then
        export MPK_FORCE_ALLGATHER_REDUCE=1
        export MPK_DISABLE_AG_OVERLAP=1
    else
        unset MPK_FORCE_ALLGATHER_REDUCE
        unset MPK_DISABLE_AG_OVERLAP
    fi

    # Per-bs MPK megakernel batched-token budget (matches the wider AE sweep
    # convention: max(8, bs) — 8 for bs<=8, bs itself when bs>8).
    local mbt
    if (( bs > 8 )); then mbt="$bs"; else mbt=8; fi

    echo "===== Fig.13  ${tag}  bs=${bs}  world=${WORLD_SIZE}  mbt=${mbt}  =====" | tee "$log"
    echo "  MPK_FORCE_ALLGATHER_REDUCE=${MPK_FORCE_ALLGATHER_REDUCE}" | tee -a "$log"
    echo "  MPK_DISABLE_AG_OVERLAP=${MPK_DISABLE_AG_OVERLAP:-<unset>}" | tee -a "$log"

    set +e
    mpirun --allow-run-as-root -np "$WORLD_SIZE" \
        "${MPIRUN_ENVS[@]}" \
        python "$SCRIPT" \
        --use-mirage \
        --max-num-batched-tokens "$mbt" \
        --model "$HF_ID" \
        --max-num-batched-requests "$bs" \
        --max-seq-length "$MAX_SEQ_LEN" \
        --ignore-eos \
        2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: ${tag} bs=${bs}  (exit $rc)" >&2
        tail -20 "$log" >&2
        return 1
    fi

    local lat
    lat=$(parse_latency_ms "$log" || true)
    if [[ -z "${lat:-}" ]]; then
        echo "Could not parse latency from $log" >&2
        return 1
    fi
    python3 - <<EOF >"$json"
import json
print(json.dumps({
    "system": "tgx",
    "experiment": "fig13_compute_comm_overlap",
    "mode": "$mode",
    "gpu": "$GPU",
    "world_size": $WORLD_SIZE,
    "model": "$HF_ID",
    "model_tag": "$MODEL_TAG",
    "script": "$SCRIPT",
    "batch_size": $bs,
    "prompt_len": $PROMPT_LEN,
    "gen_len": $GEN_LEN,
    "max_seq_length": $MAX_SEQ_LEN,
    "mpk_force_allgather_reduce": $( [[ "$mode" == "no-overlap" ]] && echo True || echo False ),
    "mpk_disable_ag_overlap": $( [[ "$mode" == "no-overlap" ]] && echo True || echo False ),
    "latency_ms_per_token": $lat,
}, indent=2))
EOF
}

for mode in $MODES; do
    for bs in $BATCH_SIZES; do
        run_cell "$mode" "$bs" || true
    done
done

echo "Fig.13 ablation sweep done. Per-cell JSONs in $OUTPUT_ROOT"

# One-shot: produce the figure straight from the JSONs we just wrote.
PLOT_SCRIPT="$MIRAGE_HOME/artifact_evaluation/H100/plot_fig13.py"
if [[ -f "$PLOT_SCRIPT" ]]; then
    echo "Plotting Fig.13 ..."
    python3 "$PLOT_SCRIPT" --input-dir "$OUTPUT_ROOT" || \
        echo "WARN: plot_fig13.py failed (you can rerun it manually)." >&2
fi
