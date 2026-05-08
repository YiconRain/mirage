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
Local prereq: `~/.ssh/id_rsa.pub` exists (run `ssh-keygen` if not).

Pick a GPU and start the box:

    modal run scripts/ae/ae_ssh.py --gpu h100
    modal run scripts/ae/ae_ssh.py --gpu a100-80gb
    modal run scripts/ae/ae_ssh.py --gpu b200
    modal run scripts/ae/ae_ssh.py --gpu h100x4
    modal run scripts/ae/ae_ssh.py --gpu h100x8

The script prints a public host:port like:

    SSH ready:  ssh root@r447.modal.host -p 45333

Copy-paste the `ssh ...` line into another terminal. Once inside, do
whatever you need (git clone, pip install, run benchmarks).

When you are done, Ctrl-C the `modal run` terminal to release the GPU
(the container will idle until its 24-hour timeout otherwise).
"""

import os
import socket
import threading
import time

import modal

APP_NAME = "tgx-osdi26-ae-ssh"

ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")

# Bare GPU container with sshd. No MPK, no torch, no clone.
# After ssh, the user installs/builds whatever they need.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install("openssh-server", "git", "curl", "build-essential")
    .run_commands(
        "mkdir -p /run/sshd",
        "mkdir -p /root/.ssh",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
    )
    .add_local_file(ssh_key_path, "/root/.ssh/authorized_keys", copy=True)
    .run_commands("chmod 600 /root/.ssh/authorized_keys")
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
    """Forward port 22 through Modal, publish address, run sshd in foreground.

    Spawns a background thread that calls results_vol.commit() every 2 minutes
    so that writes to /mirage/results survive ungraceful container exits
    (OOM kills, modal-run Ctrl-C, etc.). Without this, Modal only commits
    on clean function exit, and we have lost results before.
    """
    import subprocess

    def _commit_loop():
        while True:
            time.sleep(120)
            try:
                results_vol.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"[ae_ssh] periodic commit failed: {exc}", flush=True)

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(
            target=_wait_for_ssh, args=(host, port, q), daemon=True
        ).start()
        threading.Thread(target=_commit_loop, daemon=True).start()
        subprocess.run(["/usr/sbin/sshd", "-D"])


# ---------- TGX-image SSH entry points (one per GPU config) ----------
TIMEOUT = 3600 * 24
# Mirage's nvcc compile of search.cc + others uses ~16-24 GB of host RAM.
# Modal's default container memory is too small; bump to 64 GB.
MEMORY_MB = 64 * 1024


@app.function(gpu="A100-40GB", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_a100_40gb(q): _serve(q)


@app.function(gpu="A100-80GB", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_a100_80gb(q): _serve(q)


@app.function(gpu="H100", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_h100(q): _serve(q)


@app.function(gpu="H100:4", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_h100x4(q): _serve(q)


@app.function(gpu="H100:8", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_h100x8(q): _serve(q)


@app.function(gpu="B200", timeout=TIMEOUT, memory=MEMORY_MB)
def ssh_b200(q): _serve(q)


# ---------- Local entrypoint: spawn function, print connection info ----------
def _local_main(spawn):
    """Run a chosen ssh_* function in the cloud and print its public host:port.

    If the cloud container is restarted (e.g. OOM kill), the new container's
    address is also picked up from the queue and re-printed.
    """
    with modal.Queue.ephemeral() as q:
        spawn(q)
        try:
            while True:
                # block forever waiting for the *next* SSH address; if a
                # container restarts, the new one publishes a fresh entry.
                host, port = q.get()
                print()
                print(f"SSH ready:  ssh root@{host} -p {port}")
                print()
                print("Container will idle until you Ctrl-C this terminal")
                print("or hit the 24h timeout.")
        except KeyboardInterrupt:
            print("\nReleasing GPU.")


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
