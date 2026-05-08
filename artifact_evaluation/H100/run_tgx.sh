#!/usr/bin/env bash
# E1 single-H100 sweep, TGX/MPK system.
#
# Hopper-tuned demo entry points:
#   Qwen3-{0.6B,1.7B,8B}   -> demo/qwen3/demo_hopper.py
#   Qwen3-30B-A3B          -> demo/qwen3/demo_30B_A3B_hopper.py
#   Llama-3.2-1B-Instruct  -> demo/llama3/demo.py
#
# Each cell is parsed from stdout (the Hopper demos do not support
# --save-tokens) and written to:
#   results/H100/tgx/<model_tag>__bs<bs>.json
#
# Usage on Modal:
#   modal run scripts/ae/ae_modal.py::run_h100 \
#       --cmd "bash artifact_evaluation/H100/run_tgx.sh"
#
# Override with env vars:
#   MODELS="qwen3-0.6b qwen3-1.7b" BATCH_SIZES="1 4" bash run_tgx.sh

set -euo pipefail

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/H100/tgx}"
mkdir -p "$OUTPUT_ROOT"

MODELS="${MODELS:-qwen3-0.6b llama-3.2-1b qwen3-1.7b qwen3-8b qwen3-30b-a3b}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + GEN_LEN))}"

# (entry script, HuggingFace ID) per model tag.
declare -A SCRIPT
SCRIPT[qwen3-0.6b]="demo/qwen3/demo_hopper.py"
SCRIPT[llama-3.2-1b]="demo/llama3/demo.py"
SCRIPT[qwen3-1.7b]="demo/qwen3/demo_hopper.py"
SCRIPT[qwen3-8b]="demo/qwen3/demo_hopper.py"
SCRIPT[qwen3-30b-a3b]="demo/qwen3/demo_30B_A3B_hopper.py"

declare -A HF_ID
HF_ID[qwen3-0.6b]="Qwen/Qwen3-0.6B"
HF_ID[llama-3.2-1b]="meta-llama/Llama-3.2-1B-Instruct"
HF_ID[qwen3-1.7b]="Qwen/Qwen3-1.7B"
HF_ID[qwen3-8b]="Qwen/Qwen3-8B"
HF_ID[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"

# Parse "per-token latency ... <float> ms" from a log file -> float ms or "".
parse_latency_ms() {
    grep -oE 'per-token latency[^0-9]*[0-9]+\.[0-9]+ ms' "$1" \
        | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
}

run_cell() {
    local model_tag="$1"
    local bs="$2"
    local script="${SCRIPT[$model_tag]}"
    local hfid="${HF_ID[$model_tag]}"
    local log="$OUTPUT_ROOT/${model_tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${model_tag}__bs${bs}.json"

    echo "===== TGX  ${model_tag}  bs=${bs}  =====" | tee "$log"

    if ! python "$script" \
            --use-mirage \
            --model "$hfid" \
            --max-num-batched-requests "$bs" \
            --max-num-batched-tokens 8 \
            --max-seq-length "$MAX_SEQ_LEN" \
            --ignore-eos \
            >>"$log" 2>&1; then
        echo "FAILED: TGX ${model_tag} bs=${bs}" >&2
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
    "model": "$hfid",
    "model_tag": "$model_tag",
    "script": "$script",
    "batch_size": $bs,
    "prompt_len": $PROMPT_LEN,
    "gen_len": $GEN_LEN,
    "max_seq_length": $MAX_SEQ_LEN,
    "latency_ms_per_token": $lat,
}, indent=2))
EOF
}

for model_tag in $MODELS; do
    if [[ -z "${SCRIPT[$model_tag]:-}" ]]; then
        echo "Unknown model tag: $model_tag" >&2
        exit 1
    fi
    for bs in $BATCH_SIZES; do
        run_cell "$model_tag" "$bs" || true
    done
done

echo "TGX sweep done. Per-cell JSONs in $OUTPUT_ROOT"
