#!/usr/bin/env bash
# Fig. 11 — Qwen3-1.7B multi-GPU comparison across systems.
#
# Paper figure: relative performance of PyTorch, vLLM, SGLang, and MPK on
# Qwen3-1.7B under TP={2,4,8}, batch sizes 1/2/4/8/16, normalized to MPK
# (higher is better). Paper-matching artifact for H100; on B200 hosts
# override GPU=B200 to retag the JSONs.
#
# Per-system invocation:
#   pytorch : mpirun -np $TP python demo/qwen3/demo.py (no --use-mirage)
#             - falls into the vanilla HF decode loop; TP comes from
#               Qwen3ShardLoader collectives
#             - demo.py forces total_num_requests=1 in this path, so the
#               PyTorch curve is effectively single-request across the
#               batch-size sweep (matches Fig 9 PyTorch behavior).
#   vllm    : `vllm bench latency --tensor-parallel-size $TP ...`
#             - vLLM manages its own worker procs; no mpirun.
#   sglang  : `python -m sglang.bench_one_batch --tensor-parallel-size $TP ...`
#             - same: SGLang manages its own workers.
#   mpk     : mpirun -np $TP python demo/qwen3/demo.py --use-mirage ...
#             - uses MPK's persistent kernel + nvshmem-allreduce path.
#
# Output: results/H100/fig11/<system>__tp<TP>__bs<bs>.json
# Plot:   python artifact_evaluation/H100/plot_fig11.py
#
# Prerequisites:
#   - $TP idle GPUs visible via CUDA_VISIBLE_DEVICES.
#   - vllm in $VLLM_VENV  (default $HOME/vllm-venv)
#   - sglang in $SGLANG_VENV (default $HOME/sglang-venv)
#   - mirage_2 conda env activated when invoking this script (used by
#     pytorch + mpk systems).
#   - HF cache contains Qwen/Qwen3-1.7B (HF_HOME defaults to
#     /raid/catalyst/models on this machine; override as needed).

set -euo pipefail

export PATH="${CUDA_BIN:-/usr/local/cuda/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MIRAGE_HOME="${MIRAGE_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$MIRAGE_HOME"

GPU="${GPU:-H100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MIRAGE_HOME/results/$GPU/fig11}"
mkdir -p "$OUTPUT_ROOT"

export HF_HOME="${HF_HOME:-/raid/catalyst/models}"

# LD_PRELOAD for MPK only — see run_fig13_overlap.sh for context.
MPK_LD_PRELOAD=""
if [[ -n "${NVSHMEM_LIB_PATH:-}" && -f "${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3" ]]; then
    MPK_LD_PRELOAD="${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3"
fi

MODEL_TAG="${MODEL_TAG:-qwen3-1.7b}"
HF_ID="${HF_ID:-Qwen/Qwen3-1.7B}"
SCRIPT="${SCRIPT:-demo/qwen3/demo.py}"

TP_SIZES="${TP_SIZES:-2 4 8}"        # paper subplots; set "2 4" if only 4 GPUs are free
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((PROMPT_LEN + GEN_LEN))}"
INPUT_LEN="${INPUT_LEN:-$PROMPT_LEN}"
OUTPUT_LEN="${OUTPUT_LEN:-$GEN_LEN}"

SYSTEMS="${SYSTEMS:-pytorch vllm sglang mpk}"

VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"
SGLANG_VENV="${SGLANG_VENV:-$HOME/sglang-venv}"

# Path to the mirage_2 conda env's python and mpirun. We capture them
# now (before activating baseline venvs) so we can switch back.
MPK_PYTHON="${MPK_PYTHON:-$(command -v python)}"
MPK_MPIRUN="${MPK_MPIRUN:-$(command -v mpirun)}"
if [[ -z "$MPK_PYTHON" || -z "$MPK_MPIRUN" ]]; then
    echo "ERROR: could not locate python/mpirun in PATH; activate mirage_2 first." >&2
    exit 1
fi

parse_demo_latency_ms() {
    grep -oiE 'per-token latency[^0-9]*[0-9]+\.[0-9]+ ms' "$1" \
        | tail -1 \
        | grep -oE '[0-9]+\.[0-9]+'
}

parse_vllm_avg_s() {
    grep -oE 'Avg latency: [0-9]+\.[0-9]+' "$1" | tail -1 | awk '{print $3}'
}

parse_sglang_total_s() {
    # Try total latency line; fall back to prefill+decode sum.
    local total
    total=$(grep -oiE 'total\.? latency:?\s*[0-9]+\.[0-9]+' "$1" | tail -1 \
                | grep -oE '[0-9]+\.[0-9]+')
    if [[ -n "$total" ]]; then
        echo "$total"; return
    fi
    local prefill decode
    prefill=$(grep -oiE 'prefill\.? latency:?\s*[0-9]+\.[0-9]+' "$1" | tail -1 \
                | grep -oE '[0-9]+\.[0-9]+')
    decode=$(grep -oiE 'decode\.? latency:?\s*[0-9]+\.[0-9]+' "$1" | tail -1 \
                | grep -oE '[0-9]+\.[0-9]+')
    if [[ -n "$prefill" && -n "$decode" ]]; then
        python3 -c "print($prefill + $decode)"
    fi
}

# Convert various latency forms to the canonical latency_ms_per_token.
# Args: $1 = system, $2 = log path, $3 = batch_size
to_ms_per_token() {
    local system="$1" log="$2" bs="$3"
    case "$system" in
        pytorch|mpk)
            parse_demo_latency_ms "$log"
            ;;
        vllm)
            local avg_s
            avg_s=$(parse_vllm_avg_s "$log") || true
            if [[ -n "$avg_s" ]]; then
                python3 -c "print(($avg_s * 1000.0) / (($INPUT_LEN + $OUTPUT_LEN) * $bs))"
            fi
            ;;
        sglang)
            local total_s
            total_s=$(parse_sglang_total_s "$log") || true
            if [[ -n "$total_s" ]]; then
                python3 -c "print(($total_s * 1000.0) / (($INPUT_LEN + $OUTPUT_LEN) * $bs))"
            fi
            ;;
    esac
}

write_json() {
    local system="$1" tp="$2" bs="$3" lat="$4" json="$5"
    python3 - <<EOF >"$json"
import json
print(json.dumps({
    "experiment": "fig11_qwen3_1.7b_multi_gpu",
    "system": "$system",
    "gpu": "$GPU",
    "tp": $tp,
    "world_size": $tp,
    "model": "$HF_ID",
    "model_tag": "$MODEL_TAG",
    "batch_size": $bs,
    "input_len": $INPUT_LEN,
    "output_len": $OUTPUT_LEN,
    "prompt_len": $PROMPT_LEN,
    "gen_len": $GEN_LEN,
    "max_seq_length": $MAX_SEQ_LEN,
    "latency_ms_per_token": $lat,
}, indent=2))
EOF
}

run_cell() {
    local system="$1" tp="$2" bs="$3"
    local tag="${system}__tp${tp}"
    local log="$OUTPUT_ROOT/${tag}__bs${bs}.log"
    local json="$OUTPUT_ROOT/${tag}__bs${bs}.json"

    echo "===== Fig.11  ${system}  tp=${tp}  bs=${bs}  =====" | tee "$log"

    case "$system" in
        pytorch)
            set +e
            LD_PRELOAD="$MPK_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" \
            "$MPK_MPIRUN" --allow-run-as-root -np "$tp" \
                -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
                -x MPI_INC_PATH -x MPI_LIB_PATH \
                -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH -x HF_HOME \
                "$MPK_PYTHON" "$SCRIPT" \
                --model "$HF_ID" \
                --max-num-batched-requests "$bs" \
                --max-seq-length "$MAX_SEQ_LEN" \
                --ignore-eos \
                2>&1 | tee -a "$log"
            local rc=${PIPESTATUS[0]}
            set -e
            ;;
        mpk)
            set +e
            LD_PRELOAD="$MPK_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" \
            "$MPK_MPIRUN" --allow-run-as-root -np "$tp" \
                -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
                -x MPI_INC_PATH -x MPI_LIB_PATH \
                -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH -x HF_HOME \
                "$MPK_PYTHON" "$SCRIPT" \
                --use-mirage \
                --model "$HF_ID" \
                --max-num-batched-requests "$bs" \
                --max-seq-length "$MAX_SEQ_LEN" \
                --ignore-eos \
                2>&1 | tee -a "$log"
            local rc=${PIPESTATUS[0]}
            set -e
            ;;
        vllm)
            if [[ ! -f "$VLLM_VENV/bin/activate" ]]; then
                echo "FAILED: vllm venv missing: $VLLM_VENV" | tee -a "$log" >&2
                return 1
            fi
            set +e
            ( source "$VLLM_VENV/bin/activate"
              vllm bench latency \
                  --model "$HF_ID" \
                  --tensor-parallel-size "$tp" \
                  --input-len "$INPUT_LEN" \
                  --output-len "$OUTPUT_LEN" \
                  --batch-size "$bs" \
                  --dtype bfloat16 \
                  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.5}" \
                  --num-iters-warmup 2 \
                  --num-iters 5 ) 2>&1 | tee -a "$log"
            local rc=${PIPESTATUS[0]}
            set -e
            ;;
        sglang)
            if [[ ! -f "$SGLANG_VENV/bin/activate" ]]; then
                echo "FAILED: sglang venv missing: $SGLANG_VENV" | tee -a "$log" >&2
                return 1
            fi
            set +e
            ( source "$SGLANG_VENV/bin/activate"
              python -m sglang.bench_one_batch \
                  --model-path "$HF_ID" \
                  --tensor-parallel-size "$tp" \
                  --batch-size "$bs" \
                  --input "$INPUT_LEN" \
                  --output "$OUTPUT_LEN" \
                  --dtype bfloat16 \
                  --mem-fraction-static "${SGLANG_MEM_FRAC:-0.5}" ) 2>&1 | tee -a "$log"
            local rc=${PIPESTATUS[0]}
            set -e
            ;;
        *)
            echo "Unknown system: $system" >&2; return 1
            ;;
    esac

    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED: ${system} tp=${tp} bs=${bs}  (exit $rc)" >&2
        tail -20 "$log" >&2
        return 1
    fi

    local lat
    lat=$(to_ms_per_token "$system" "$log" "$bs" || true)
    if [[ -z "${lat:-}" ]]; then
        echo "Could not parse latency from $log" >&2
        return 1
    fi
    write_json "$system" "$tp" "$bs" "$lat" "$json"
}

echo "Fig.11 sweep: systems=[$SYSTEMS]  tp=[$TP_SIZES]  bs=[$BATCH_SIZES]"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"

for tp in $TP_SIZES; do
    for system in $SYSTEMS; do
        for bs in $BATCH_SIZES; do
            run_cell "$system" "$tp" "$bs" || true
        done
    done
done

echo "Fig.11 sweep done. Per-cell JSONs in $OUTPUT_ROOT"

# One-shot: produce the figure straight from the JSONs we just wrote.
PLOT_SCRIPT="$MIRAGE_HOME/artifact_evaluation/H100/plot_fig11.py"
if [[ -f "$PLOT_SCRIPT" ]]; then
    echo "Plotting Fig.11 ..."
    "$MPK_PYTHON" "$PLOT_SCRIPT" --input-dir "$OUTPUT_ROOT" || \
        echo "WARN: plot_fig11.py failed (you can rerun it manually)." >&2
fi
