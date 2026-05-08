"""
SSH-able Modal containers for OSDI '26 TGX artifact evaluation.

Same image as scripts/ae/ae_modal.py, but instead of running a one-shot
--cmd, each entry point starts an sshd on port 22, forwards it through a
Modal tunnel to a local port, and idles for up to 24 hours. You ssh in
and drive the artifact_evaluation/<gpu>/*.sh scripts directly.

Why SSH
-------
- Same bash scripts work on Modal *and* any other GPU host (on-prem,
  Lambda, Crusoe, etc.). Nothing in artifact_evaluation/ depends on
  Modal-specific env.
- Interactive: re-run failed cells, edit scripts in place, set HF_TOKEN
  for the session, watch nvidia-smi, etc.
- Persistent: the tgx-ae-results Volume holds your JSONs across restarts.

Usage
-----
Local prereqs (one-time):

    pip install sshtunnel
    # Make sure ~/.ssh/id_rsa.pub exists; ssh-keygen if not.

Pick a GPU and start the box:

    modal run scripts/ae/ae_ssh.py::ssh_h100
    modal run scripts/ae/ae_ssh.py::ssh_a100_80gb
    modal run scripts/ae/ae_ssh.py::ssh_b200
    modal run scripts/ae/ae_ssh.py::ssh_h100x4
    modal run scripts/ae/ae_ssh.py::ssh_h100x8

The script prints something like:

    SSH server running at <host>:<port>
    SSH tunnel forwarded to localhost:9090

In another terminal:

    ssh -p 9090 root@localhost
    cd /mirage
    git pull --quiet
    bash artifact_evaluation/H100/run_tgx.sh

When you are done, Ctrl-C the local entrypoint to tear down the tunnel
(the container will idle until its 24-hour timeout).

Use the baseline image variants (`ssh_*_baselines`) when you want to
run vLLM / SGLang sweeps; that image has them pre-installed but no MPK.
"""

import os
import socket
import threading
import time

import modal

APP_NAME = "tgx-osdi26-ae-ssh"
BRANCH = "tgx-osdi26-ae"
LOCAL_PORT = 9090

ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")

# ---------- TGX image (mirrors ae_modal.py + sshd) ----------
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
        "openssh-server",
    )
    .run_commands(
        "mkdir -p /run/sshd",
        "mkdir -p /root/.ssh",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
    )
    .add_local_file(ssh_key_path, "/root/.ssh/authorized_keys", copy=True)
    .run_commands("chmod 600 /root/.ssh/authorized_keys")
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
    .pip_install("torch==2.6.0", "transformers", "mpi4py")
    .run_commands("cd mirage && pip install -e . -v")
    .run_commands(
        "pip install flashinfer-python "
        "-i https://flashinfer.ai/whl/cu124/torch2.6"
    )
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
def _wait_for_ssh(host, port, q):
    """Poll local ssh port until it accepts connections, then publish (host, port)."""
    start = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.05)
            if time.monotonic() - start >= 60.0:
                raise TimeoutError("port 22 never opened") from exc
        q.put((host, port))


def _serve(q):
    """Forward port 22 through Modal, publish address, run sshd in foreground."""
    import subprocess

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(
            target=_wait_for_ssh, args=(host, port, q), daemon=True
        ).start()
        subprocess.run(["/usr/sbin/sshd", "-D"])


# ---------- TGX-image SSH entry points (one per GPU config) ----------
TIMEOUT = 3600 * 24


@app.function(gpu="A100-40GB", timeout=TIMEOUT)
def ssh_a100_40gb(q): _serve(q)


@app.function(gpu="A100-80GB", timeout=TIMEOUT)
def ssh_a100_80gb(q): _serve(q)


@app.function(gpu="H100", timeout=TIMEOUT)
def ssh_h100(q): _serve(q)


@app.function(gpu="H100:4", timeout=TIMEOUT)
def ssh_h100x4(q): _serve(q)


@app.function(gpu="H100:8", timeout=TIMEOUT)
def ssh_h100x8(q): _serve(q)


@app.function(gpu="B200", timeout=TIMEOUT)
def ssh_b200(q): _serve(q)


# ---------- Local entrypoint: spawn function, set up tunnel ----------
def _local_main(spawn):
    """Run a chosen ssh_* function in the cloud and forward to LOCAL_PORT."""
    import sshtunnel

    with modal.Queue.ephemeral() as q:
        spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")

        server = sshtunnel.SSHTunnelForwarder(
            (host, port),
            ssh_username="root",
            ssh_password="password",  # bypassed by pubkey auth
            remote_bind_address=("127.0.0.1", 22),
            local_bind_address=("127.0.0.1", LOCAL_PORT),
            allow_agent=False,
        )
        try:
            server.start()
            print(f"SSH tunnel forwarded to localhost:{server.local_bind_port}")
            print(f"Connect:  ssh -p {server.local_bind_port} root@localhost")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SSH tunnel...")
        finally:
            server.stop()


@app.local_entrypoint()
def main(gpu: str = "h100"):
    """Default local entrypoint. Pick the GPU with --gpu.

    Examples:
        modal run scripts/ae/ae_ssh.py
        modal run scripts/ae/ae_ssh.py --gpu h100
        modal run scripts/ae/ae_ssh.py --gpu a100-80gb
        modal run scripts/ae/ae_ssh.py --gpu b200
        modal run scripts/ae/ae_ssh.py --gpu h100x4
        modal run scripts/ae/ae_ssh.py --gpu h100x4-baselines
    """
    table = {
        "a100-40gb": ssh_a100_40gb,
        "a100-80gb": ssh_a100_80gb,
        "h100": ssh_h100,
        "h100x4": ssh_h100x4,
        "h100x8": ssh_h100x8,
        "b200": ssh_b200,
    }
    fn = table.get(gpu.lower())
    if fn is None:
        raise SystemExit(f"unknown --gpu {gpu!r}; choices: {sorted(table)}")
    _local_main(fn.spawn)
