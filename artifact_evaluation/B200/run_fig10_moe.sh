#!/usr/bin/env bash
# Fig. 10 — Qwen3-30B-A3B MoE microbenchmark on a single B200.
#
# The paper plots MoE-block runtime (in μs) for three configurations:
#   - SGLang-MoE       : SGLang's MoE implementation
#   - MPK-Static-MoE   : MPK with static per-expert SM assignment (ablation
#                        baseline, NOT exposed in this AE branch — used for
#                        internal justification of the hybrid balancer)
#   - MPK-Hybrid-MoE   : MPK with the hybrid workload balancer + fused
#                        gather-GEMM (the production setting; what this
#                        artifact reproduces)
#
# This script runs the MPK-Hybrid-MoE configuration (TGX) and emits
# per-token end-to-end latency across batch sizes 1..16. Qwen3-30B-A3B is
# dominated by its MoE blocks, so the per-token latency tracks MoE-only
# runtime closely. For the per-MoE-block μs numbers in the paper, run an
# NCU profile of the moe_w13_linear / moe_w2_linear kernels — see README
# §"Fig. 10".
#
# The SGLang-MoE bar is produced by `run_sglang.sh MODELS=qwen3-30b-a3b`.
# Compare the two outputs to reproduce Fig. 10's MPK-Hybrid vs SGLang
# comparison.
#
# Note: the demo's MAX_TOKENS = 1 attention quirk that previously required
# a manual header edit is now handled automatically at compile time by
# `task_register.cc` (see commit 688632e), based on the model's GQA ratio.
# No manual rebuild needed for this script.
#
# Output: results/B200/fig10/<system>__bs<bs>.json
# See artifact_evaluation/B200/README.md for the SGLang baseline.

set -euo pipefail

export PATH="${CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/B200/fig10}"
mkdir -p "$OUTPUT_ROOT"

MODEL_TAG="${MODEL_TAG:-qwen3-30b-a3b}"
HF_ID="${HF_ID:-Qwen/Qwen3-30B-A3B}"
SCRIPT="${SCRIPT:-demo/qwen3/demo_30B_A3B.py}"

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + GEN_LEN))}"

parse_latency_ms() {
    grep -oiE 'per-token latency[^0-9]*[0-9]+\.[0-9]+ ms' "$1" \
        | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
}

run_cell() {
    local bs="$1"
    local tag="tgx-hybrid-moe"
    local log="$OUTPUT_ROOT/${tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${tag}__bs${bs}.json"

    echo "===== Fig.10  ${tag}  bs=${bs}  =====" | tee "$log"

    set +e
    python "$SCRIPT" \
        --use-mirage \
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
    "experiment": "fig10_moe",
    "variant": "hybrid",
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

for bs in $BATCH_SIZES; do
    run_cell "$bs" || true
done

echo "Fig.10 MoE sweep done. Per-cell JSONs in $OUTPUT_ROOT"
echo "For the SGLang-MoE bar, run: bash artifact_evaluation/B200/run_sglang.sh"
echo "    with MODELS=qwen3-30b-a3b and compare per-token latencies."
