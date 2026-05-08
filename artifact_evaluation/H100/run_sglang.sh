#!/usr/bin/env bash
# E1 single-H100 sweep, SGLang baseline.
#
# Output: results/H100/sglang/<model_tag>__bs<bs>.{log,json}
#
# Run inside the BASELINES Modal image:
#   modal run scripts/ae/ae_modal.py::baseline_h100 \
#       --cmd "bash artifact_evaluation/H100/run_sglang.sh"

set -euo pipefail

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
cd "$MIRAGE_HOME"

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/H100/sglang}"
mkdir -p "$OUTPUT_ROOT"

MODELS="${MODELS:-qwen3-0.6b llama-3.2-1b qwen3-1.7b qwen3-8b qwen3-30b-a3b}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
INPUT_LEN="${INPUT_LEN:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"

declare -A HF_ID
HF_ID[qwen3-0.6b]="Qwen/Qwen3-0.6B"
HF_ID[llama-3.2-1b]="meta-llama/Llama-3.2-1B-Instruct"
HF_ID[qwen3-1.7b]="Qwen/Qwen3-1.7B"
HF_ID[qwen3-8b]="Qwen/Qwen3-8B"
HF_ID[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"

run_cell() {
    local model_tag="$1"
    local bs="$2"
    local hfid="${HF_ID[$model_tag]}"
    local log="$OUTPUT_ROOT/${model_tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${model_tag}__bs${bs}.json"

    echo "===== SGLANG  ${model_tag}  bs=${bs}  =====" | tee "$log"

    # bench_offline_throughput accepts --num-prompts (= batch size in our
    # offline-batched setting), --input-length, --output-length.
    if ! python -m sglang.bench_offline_throughput \
            --model "$hfid" \
            --num-prompts "$bs" \
            --input-length "$INPUT_LEN" \
            --output-length "$OUTPUT_LEN" \
            --dtype bfloat16 \
            >>"$log" 2>&1; then
        echo "FAILED: sglang ${model_tag} bs=${bs}" >&2
        return 1
    fi

    # SGLang prints e.g. "Token generation throughput: X.XX token/s"
    # We invert to ms/token.
    local tgen
    tgen=$(grep -oE 'generation throughput[^0-9]*[0-9]+\.[0-9]+' "$log" \
            | tail -1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ -z "${tgen:-}" ]]; then
        echo "Could not parse throughput from $log" >&2
        return 1
    fi
    python3 - <<EOF >"$json"
import json
tps = $tgen
ms_per_token = 1000.0 / tps if tps > 0 else float("nan")
print(json.dumps({
    "system": "sglang",
    "model": "$hfid",
    "batch_size": $bs,
    "input_len": $INPUT_LEN,
    "output_len": $OUTPUT_LEN,
    "tokens_per_second": tps,
    "latency_ms_per_token": ms_per_token,
}, indent=2))
EOF
}

for model_tag in $MODELS; do
    if [[ -z "${HF_ID[$model_tag]:-}" ]]; then
        echo "Unknown model tag: $model_tag" >&2
        exit 1
    fi
    for bs in $BATCH_SIZES; do
        run_cell "$model_tag" "$bs" || true
    done
done

echo "SGLang sweep done. Per-cell JSONs in $OUTPUT_ROOT"
