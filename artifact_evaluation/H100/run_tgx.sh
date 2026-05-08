#!/usr/bin/env bash
# E1 single-H100 sweep, TGX/MPK system.
#
# Runs each model × batch size in {1,2,4,8,16} and writes per-cell JSON
# (containing latency_ms_per_token and a snippet of generated text) to:
#   results/H100/tgx/<model_tag>__bs<bs>.json
#
# Run inside the TGX Modal image:
#   modal run scripts/ae/ae_modal.py::run_h100 \
#       --cmd "bash artifact_evaluation/H100/run_tgx.sh"
#
# Override the model/bs lists with env vars:
#   MODELS="qwen3-0.6b qwen3-1.7b" BATCH_SIZES="1 4" bash run_tgx.sh

set -euo pipefail

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/H100/tgx}"
mkdir -p "$OUTPUT_ROOT"

# Default models — paper Fig. 9 row 2 (H100), all 5 models.
MODELS="${MODELS:-qwen3-0.6b llama-3.2-1b qwen3-1.7b qwen3-8b qwen3-30b-a3b}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# Map our short tag → (entry script, HuggingFace ID).
declare -A SCRIPT
SCRIPT[qwen3-0.6b]="demo/qwen3/demo.py"
SCRIPT[llama-3.2-1b]="demo/llama3/demo.py"
SCRIPT[qwen3-1.7b]="demo/qwen3/demo.py"
SCRIPT[qwen3-8b]="demo/qwen3/demo.py"
SCRIPT[qwen3-30b-a3b]="demo/qwen3/demo_30B_A3B.py"

declare -A HF_ID
HF_ID[qwen3-0.6b]="Qwen/Qwen3-0.6B"
HF_ID[llama-3.2-1b]="meta-llama/Llama-3.2-1B-Instruct"
HF_ID[qwen3-1.7b]="Qwen/Qwen3-1.7B"
HF_ID[qwen3-8b]="Qwen/Qwen3-8B"
HF_ID[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"

run_cell() {
    local model_tag="$1"
    local bs="$2"
    local script="${SCRIPT[$model_tag]}"
    local hfid="${HF_ID[$model_tag]}"
    local out="$OUTPUT_ROOT/${model_tag}__bs${bs}.json"
    local log="$OUTPUT_ROOT/${model_tag}__bs${bs}.log"

    echo "===== TGX  ${model_tag}  bs=${bs}  =====" | tee -a "$log"

    python "$script" \
        --use-mirage \
        --model "$hfid" \
        --max-num-batched-requests "$bs" \
        --max-num-batched-tokens 8 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0 \
        --save-tokens "$out" \
        2>&1 | tee -a "$log"
}

for model_tag in $MODELS; do
    if [[ -z "${SCRIPT[$model_tag]:-}" ]]; then
        echo "Unknown model tag: $model_tag" >&2
        exit 1
    fi
    for bs in $BATCH_SIZES; do
        if ! run_cell "$model_tag" "$bs"; then
            echo "FAILED: TGX ${model_tag} bs=${bs}" >&2
            # keep going so partial sweeps still produce some data
        fi
    done
done

echo "TGX sweep done. Per-cell JSONs in $OUTPUT_ROOT"
