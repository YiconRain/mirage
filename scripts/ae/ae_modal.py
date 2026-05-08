"""
Modal container for OSDI '26 TGX artifact evaluation.

Builds an image off the `tgx-osdi26-ae` branch of mirage, mounts a persistent
HuggingFace cache volume, and exposes one function per GPU configuration
needed by the evaluation matrix.

Usage
-----
Run an arbitrary shell command inside a freshly provisioned container:

    modal run scripts/ae/ae_modal.py::run_h100 --cmd \
        "python /mirage/demo/qwen3/demo.py --use-mirage --max-num-batched-requests 1"

    modal run scripts/ae/ae_modal.py::run_a100_80gb --cmd "..."
    modal run scripts/ae/ae_modal.py::run_b200    --cmd "..."
    modal run scripts/ae/ae_modal.py::run_h100x4  --cmd "..."
    modal run scripts/ae/ae_modal.py::run_h100x8  --cmd "..."

Smoke test (no GPU work, just verifies the image builds and MPK imports):

    modal run scripts/ae/ae_modal.py::smoke_test

Notes
-----
- Image clones the `tgx-osdi26-ae` branch. To test local edits, switch the
  branch arg in BRANCH below.
- HuggingFace weights live in the `tgx-ae-hf-cache` volume so they are reused
  across runs.
- Multi-GPU runs (`run_h100x4`, `run_h100x8`) launch via `mpirun -n N` inside
  the function, so the user's --cmd should be a single Python invocation; the
  wrapper prepends mpirun.
"""

import modal

APP_NAME = "tgx-osdi26-ae"
BRANCH = "tgx-osdi26-ae"

# ---------- Image ----------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install(
        "wget",
        "sudo",
        "binutils",
        "git",
        "libmpich-dev",
        "libopenmpi-dev",
        "openmpi-bin",
        "curl",
        "pkg-config",
        "build-essential",
    )
    .run_commands(
        f"git clone --recursive --branch {BRANCH} "
        "https://github.com/mirage-project/mirage.git"
    )
    .env({"MIRAGE_HOME": "/mirage"})
    .env(
        {
            "LD_LIBRARY_PATH": (
                "/mirage/build/abstract_subexpr/release:"
                "/mirage/build/formal_verifier/release:$LD_LIBRARY_PATH"
            )
        }
    )
    .run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .run_commands(
        "cd mirage && uv pip install --system -e . -v "
        "transformers torch==2.6.0 mpi4py"
    )
    .run_commands(
        "cd mirage && uv pip install --system flashinfer-python "
        "-i https://flashinfer.ai/whl/cu124/torch2.6"
    )
    .run_commands("uv pip install --system vllm sglang")
)

hf_cache_vol = modal.Volume.from_name("tgx-ae-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("tgx-ae-results", create_if_missing=True)

app = modal.App(
    APP_NAME,
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/mirage/results": results_vol,
    },
)

# ---------- Helpers ----------
def _run(cmd: str, world_size: int = 1) -> None:
    import subprocess

    if world_size > 1:
        cmd = f"mpirun -n {world_size} --allow-run-as-root --bind-to none {cmd}"
    print(f"[ae_modal] $ {cmd}")
    subprocess.run(cmd, check=True, shell=True, executable="/bin/bash", cwd="/mirage")


# ---------- Per-GPU entry points ----------
@app.function(gpu="A100-40GB", timeout=3600 * 4)
def run_a100_40gb(cmd: str) -> None:
    _run(cmd)


@app.function(gpu="A100-80GB", timeout=3600 * 4)
def run_a100_80gb(cmd: str) -> None:
    _run(cmd)


@app.function(gpu="H100", timeout=3600 * 4)
def run_h100(cmd: str) -> None:
    _run(cmd)


@app.function(gpu="H100:4", timeout=3600 * 4)
def run_h100x4(cmd: str) -> None:
    _run(cmd, world_size=4)


@app.function(gpu="H100:8", timeout=3600 * 4)
def run_h100x8(cmd: str) -> None:
    _run(cmd, world_size=8)


@app.function(gpu="B200", timeout=3600 * 4)
def run_b200(cmd: str) -> None:
    _run(cmd)


# ---------- Smoke test ----------
@app.function(gpu="H100", timeout=600)
def smoke_test() -> None:
    """Verify the image is sane: clone present, MPK imports, GPU visible."""
    import subprocess

    subprocess.run("nvidia-smi", check=True, shell=True)
    subprocess.run(
        "python -c 'import mirage; print(\"mirage:\", mirage.__file__)'",
        check=True, shell=True, cwd="/mirage",
    )
    subprocess.run("git -C /mirage log -1 --oneline", check=True, shell=True)
    print("[ae_modal] smoke test OK")
