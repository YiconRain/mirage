#!/usr/bin/env bash
# Fig. 12 — cross-task pipelining ablation on B200.
#
# The paper plots the runtime of the final lm_head linear layer in Qwen3-8B
# with three settings:
#   - CUBLAS         : cuBLAS baseline (PyTorch / vanilla)
#   - MPK-No-Pipe    : MPK, lm_head split into too many tasks for the
#                      cross-task pipeliner to engage
#                      (vocab_size // 256 = 153600 // 256 = 600 tasks)
#   - MPK-Pipe       : MPK, lm_head split into 128 tasks so the pipeliner
#                      can overlap the next layer's pre-load with the
#                      current layer's compute (the default for production)
#
# This script runs Qwen3-8B end-to-end on TGX with both --lm-head-grid pipe
# and --lm-head-grid no-pipe across batch sizes 1, 2, 4, 8, 16, and reports
# per-token latency. The pipe-vs-no-pipe delta reflects the lm_head
# pipelining benefit. For the absolute per-layer microsecond numbers in the
# paper, run an NCU profile of the lm_head kernel — see README §"Fig. 12".
#
# Output: results/B200/fig12/<mode>__bs<bs>.json
# See artifact_evaluation/B200/README.md for the cuBLAS baseline (use
# run_pytorch.sh with the same model).

set -euo pipefail

export PATH="${CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/B200/fig12}"
mkdir -p "$OUTPUT_ROOT"

MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
HF_ID="${HF_ID:-Qwen/Qwen3-8B}"
SCRIPT="${SCRIPT:-demo/qwen3/demo.py}"

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + GEN_LEN))}"

# The two ablation modes
MODES="${MODES:-pipe no-pipe}"

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

    echo "===== Fig.12  ${tag}  bs=${bs}  =====" | tee "$log"

    set +e
    python "$SCRIPT" \
        --use-mirage \
        --model "$HF_ID" \
        --max-num-batched-requests "$bs" \
        --max-seq-length "$MAX_SEQ_LEN" \
        --lm-head-grid "$mode" \
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
    "experiment": "fig12_cross_task_pipelining",
    "mode": "$mode",
    "gpu": "B200",
    "model": "$HF_ID",
    "model_tag": "$MODEL_TAG",
    "script": "$SCRIPT",
    "batch_size": $bs,
    "prompt_len": $PROMPT_LEN,
    "gen_len": $GEN_LEN,
    "max_seq_length": $MAX_SEQ_LEN,
    "latency_ms_per_token": $lat,
}, indent=2))
EOF
}

for mode in $MODES; do
    for bs in $BATCH_SIZES; do
        run_cell "$mode" "$bs" || true
    done
done

echo "Fig.12 ablation sweep done. Per-cell JSONs in $OUTPUT_ROOT"
