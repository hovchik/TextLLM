#!/usr/bin/env python3
"""One command to set up and start Text LLM Studio.

    python run.py                    # set up (first time only) and open http://127.0.0.1:7860
    python run.py --port 8000        # another port
    python run.py --host 0.0.0.0     # reachable from other machines on your network
    python run.py --no-browser       # don't open a browser tab
    python run.py --reinstall        # reinstall the Python packages
    python run.py --gpu              # install the CUDA build of PyTorch instead of the CPU build
    python run.py --no-venv          # use the current Python instead of .venv
    python run.py --no-fast          # skip llama-cpp-python (fast quantized chat on CPU)

What it does:
  1. Creates a virtual environment in .venv (first run only).
  2. Installs requirements.txt into it. On Linux and Windows it installs the CPU build of
     PyTorch, which is much smaller than the default CUDA build. Packages are only
     reinstalled when requirements.txt changes.
  3. Tries once to install llama-cpp-python (prebuilt CPU wheel), which makes chat with
     fine-tuned models several times faster through 4-bit GGUF copies. If it can't be
     installed, everything still works, just slower.
  4. Creates the folders the app writes to (runs/, data/raw/, data/bin/).
  5. Starts the web app (app.py) and opens it in your browser. If the port is busy it
     uses the next free one.
Stop it with Ctrl+C.

This file only uses the Python standard library, so it runs before anything is installed.
"""
import argparse, hashlib, os, socket, subprocess, sys, threading, time, venv, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQS = os.path.join(ROOT, "requirements.txt")
STAMP = os.path.join(VENV, ".requirements.sha256")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
LLAMA_CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
FAST_STAMP = os.path.join(VENV, ".fast-attempted")


def say(msg):
    print(f"[run] {msg}", flush=True)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def ensure_venv():
    py = venv_python()
    if not os.path.exists(py):
        say("Creating virtual environment in .venv (first run only) ...")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as e:
            sys.exit(f"[run] Could not create .venv ({e}).\n"
                     "      On Debian/Ubuntu install it with: sudo apt install python3-venv\n"
                     "      Or skip the venv with: python run.py --no-venv")
    return py


def pip(py, *args, required=True):
    cmd = [py, "-m", "pip", "install", "--disable-pip-version-check", *args]
    ok = subprocess.call(cmd, cwd=ROOT) == 0
    if not ok and required:
        sys.exit("[run] Package install failed (see the pip output above). "
                 "Fix the error, then run this script again.")
    return ok


def requirements_met(py):
    """True if every package in requirements.txt is installed for this Python."""
    names = [l.split("#")[0].strip() for l in open(REQS)]
    code = ("import sys, importlib.metadata as m\n"
            "for n in sys.argv[1:]:\n"
            "    m.version(n.split('[')[0].split('=')[0].split('<')[0].split('>')[0].strip())")
    return subprocess.call([py, "-c", code, *[n for n in names if n]], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def torch_has_cuda_build(py):
    return subprocess.call([py, "-c", "import torch, sys; sys.exit(0 if torch.version.cuda else 1)"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def has_module(py, name):
    return subprocess.call([py, "-c", f"import {name}"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def ensure_packages(py, reinstall=False, gpu=False):
    with open(REQS, "rb") as f:
        want = hashlib.sha256(f.read() + (b"gpu" if gpu else b"cpu")).hexdigest()
    stamp = STAMP if py == venv_python() else None
    if stamp and not reinstall and os.path.exists(stamp):
        with open(stamp) as f:
            if f.read().strip() == want:
                return
    if not stamp and not reinstall and not gpu and requirements_met(py):
        return  # --no-venv and everything in requirements.txt is already there
    say("Installing Python packages (first run takes a few minutes) ...")
    pip(py, "--upgrade", "pip")
    # The default PyTorch wheel on Linux/Windows bundles CUDA (several GB). This app trains on CPU,
    # so install the CPU build unless asked otherwise. macOS wheels are already CPU/MPS only.
    if not gpu and sys.platform != "darwin" and not has_module(py, "torch"):
        if not pip(py, "torch", "--index-url", TORCH_CPU_INDEX, "--retries", "1", required=False):
            say("Could not reach the PyTorch CPU index, installing the regular PyTorch build instead.")
    elif gpu and sys.platform != "darwin" and has_module(py, "torch") and not torch_has_cuda_build(py):
        say("Replacing the CPU build of PyTorch with the CUDA build ...")
        pip(py, "--force-reinstall", "torch")
    pip(py, "-r", REQS)
    if stamp:
        with open(stamp, "w") as f:
            f.write(want)


def ensure_fast(py, skip=False):
    """Best effort: llama-cpp-python for fast GGUF chat. Tried once; never fails the setup."""
    if skip or has_module(py, "llama_cpp"):
        return
    stamp = FAST_STAMP if py == venv_python() else None
    if stamp and os.path.exists(stamp):
        return
    say("Installing llama-cpp-python for fast CPU chat (optional) ...")
    ok = pip(py, "llama-cpp-python", "--only-binary", "llama-cpp-python", "--extra-index-url", LLAMA_CPU_INDEX,
             "--retries", "1", required=False)
    if not ok:
        say("No prebuilt llama-cpp-python for this system; chat will use PyTorch (slower). "
            "To build it yourself: pip install llama-cpp-python")
    if stamp:
        open(stamp, "w").write("ok" if ok else "failed")


def ensure_dirs():
    for d in ("runs", os.path.join("data", "raw"), os.path.join("data", "bin")):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)


def port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("" if host == "0.0.0.0" else host, port))
            return True
        except OSError:
            return False


def pick_port(host, port):
    for p in range(port, port + 20):
        if port_free(host, p):
            if p != port:
                say(f"Port {port} is busy, using {p} instead.")
            return p
    sys.exit(f"[run] Ports {port}-{port + 19} are all busy. Pass another one with --port.")


def open_when_ready(url, host, port, proc):
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    for _ in range(600):  # up to ~2 minutes (first start imports torch)
        if proc.poll() is not None:
            return
        try:
            with socket.create_connection((probe, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Set up and start Text LLM Studio.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument("--reinstall", action="store_true", help="reinstall the Python packages")
    ap.add_argument("--gpu", action="store_true", help="install the default (CUDA) PyTorch build")
    ap.add_argument("--no-venv", action="store_true", help="use the current Python instead of .venv")
    ap.add_argument("--no-fast", action="store_true", help="don't install llama-cpp-python")
    a = ap.parse_args()

    if sys.version_info < (3, 9):
        sys.exit(f"[run] Python 3.9 or newer is needed (this is {sys.version.split()[0]}).")
    os.chdir(ROOT)

    py = sys.executable if a.no_venv else ensure_venv()
    ensure_packages(py, reinstall=a.reinstall, gpu=a.gpu)
    ensure_fast(py, skip=a.no_fast)
    ensure_dirs()

    port = pick_port(a.host, a.port)
    shown = "127.0.0.1" if a.host == "0.0.0.0" else a.host
    url = f"http://{shown}:{port}"
    say(f"Starting Text LLM Studio at {url}  (Ctrl+C to stop)")

    proc = subprocess.Popen([py, os.path.join(ROOT, "app.py"), "--host", a.host, "--port", str(port)], cwd=ROOT)
    if not a.no_browser:
        threading.Thread(target=open_when_ready, args=(url, a.host, port, proc), daemon=True).start()
    try:
        sys.exit(proc.wait())
    except KeyboardInterrupt:
        say("Stopping ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
