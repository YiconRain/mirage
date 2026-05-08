#!/usr/bin/env bash
# Bootstrap a bare GPU host into a state where artifact_evaluation/<gpu>/*.sh
# can run. Idempotent: re-running is safe.
#
# Usage:
#   bash artifact_evaluation/setup.sh                  # TGX/MPK install
#   bash artifact_evaluation/setup.sh --with-baselines # also install vLLM + SGLang
#   bash artifact_evaluation/setup.sh --baselines-only # only vLLM + SGLang (no MPK)
#
# Honors:
#   MIRAGE_HOME (default: /mirage)
#   BRANCH      (default: tgx-osdi26-ae)
#   HF_TOKEN    (forwarded to huggingface-cli login if set)
#
# Tested on: Ubuntu 22.04 + CUDA 12.4 (the Modal ae_ssh image, and a fresh
# Lambda H100 instance).

set -euo pipefail

MIRAGE_HOME="${MIRAGE_HOME:-/mirage}"
BRANCH="${BRANCH:-tgx-osdi26-ae}"
WITH_BASELINES=0
BASELINES_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --with-baselines)  WITH_BASELINES=1 ;;
        --baselines-only)  BASELINES_ONLY=1; WITH_BASELINES=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ---------- 1. system packages ----------
need_apt=()
for pkg in git curl build-essential pkg-config wget python3 python3-pip; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt+=("$pkg")
done
# Multi-GPU runs need MPI; harmless to install even on single-GPU hosts.
for pkg in libmpich-dev libopenmpi-dev openmpi-bin; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt+=("$pkg")
done
if (( ${#need_apt[@]} )); then
    echo "[setup] apt installing: ${need_apt[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${need_apt[@]}"
fi

# ---------- 2. mirage repo ----------
if (( ! BASELINES_ONLY )); then
    # If $MIRAGE_HOME exists but isn't a git repo (e.g. left over from a
    # stale Modal image layer), clear it before cloning.
    if [[ -d "$MIRAGE_HOME" && ! -d "$MIRAGE_HOME/.git" ]]; then
        echo "[setup] $MIRAGE_HOME exists but is not a git repo; clearing"
        rm -rf "$MIRAGE_HOME"
    fi
    if [[ ! -d "$MIRAGE_HOME/.git" ]]; then
        echo "[setup] cloning mirage@${BRANCH} into $MIRAGE_HOME"
        git clone --recursive --branch "$BRANCH" \
            https://github.com/mirage-project/mirage.git "$MIRAGE_HOME"
    else
        echo "[setup] updating $MIRAGE_HOME (branch $BRANCH)"
        git -C "$MIRAGE_HOME" fetch --quiet origin "$BRANCH"
        git -C "$MIRAGE_HOME" checkout --quiet "$BRANCH"
        git -C "$MIRAGE_HOME" reset --hard --quiet "origin/$BRANCH"
        git -C "$MIRAGE_HOME" submodule update --init --recursive --quiet
    fi
    export MIRAGE_HOME
    grep -q '^export MIRAGE_HOME=' /root/.bashrc 2>/dev/null \
        || echo "export MIRAGE_HOME=$MIRAGE_HOME" >> /root/.bashrc

    # CUDA toolchain on PATH (nvcc lives in /usr/local/cuda/bin on the
    # nvidia/cuda:*-devel base image, but PATH isn't set by default).
    if [[ -d /usr/local/cuda/bin ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export CUDA_HOME="/usr/local/cuda"
        grep -q '^export PATH=/usr/local/cuda/bin' /root/.bashrc 2>/dev/null \
            || cat >> /root/.bashrc <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
EOF
    fi

    # ---------- 3. rust (only needed for abstract_subexpr / formal_verifier) ----------
    if ! command -v rustc >/dev/null 2>&1; then
        echo "[setup] installing rust toolchain"
        curl -sSf https://sh.rustup.rs | bash -s -- -y --default-toolchain stable
    fi
    export PATH="$HOME/.cargo/bin:$PATH"

    # ---------- 4. python deps + MPK build ----------
    echo "[setup] pip install MPK + deps (this builds the C++/CUDA extension)"
    pip install --upgrade pip
    pip install torch==2.6.0 transformers mpi4py
    (cd "$MIRAGE_HOME" && pip install -e . -v)
    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6
fi

# ---------- 5. baselines (optional) ----------
if (( WITH_BASELINES )); then
    echo "[setup] pip install vLLM + SGLang (baselines)"
    # vllm and sglang each manage their own torch pin; install in a fresh
    # virtualenv to avoid clobbering the MPK environment.
    if (( BASELINES_ONLY )); then
        pip install vllm
        pip install 'sglang[all]'
    else
        BASELINES_VENV="${BASELINES_VENV:-/opt/baselines-venv}"
        if [[ ! -d "$BASELINES_VENV" ]]; then
            python3 -m venv "$BASELINES_VENV"
        fi
        # shellcheck disable=SC1091
        source "$BASELINES_VENV/bin/activate"
        pip install --upgrade pip
        pip install vllm
        pip install 'sglang[all]'
        deactivate
        echo "[setup] baselines installed in $BASELINES_VENV"
        echo "        activate before running run_vllm.sh / run_sglang.sh:"
        echo "        source $BASELINES_VENV/bin/activate"
    fi
fi

# ---------- 6. HF auth ----------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] logging in to HuggingFace"
    pip install --quiet 'huggingface_hub[cli]' || true
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
fi

# ---------- summary ----------
echo
echo "[setup] done."
if (( ! BASELINES_ONLY )); then
    echo "  MIRAGE_HOME=$MIRAGE_HOME"
    echo "  python -c 'import mirage; print(mirage.__file__)'"
fi
echo
echo "Next:"
echo "  bash $MIRAGE_HOME/artifact_evaluation/H100/run_tgx.sh"
