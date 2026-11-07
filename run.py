#!/usr/bin/env python3
"""One command to set up and start Text LLM Studio.

    python run.py                    # set up (first time only) and open http://127.0.0.1:7860
    python run.py --port 8000        # another port
    python run.py --host 0.0.0.0     # reachable from other machines on your network
    python run.py --no-browser       # don't open a browser tab
    python run.py --reinstall        # reinstall the Python packages
    python run.py --gpu              # force the CUDA (NVIDIA GPU) build of PyTorch
    python run.py --cpu              # force the CPU build of PyTorch (no GPU)
    python run.py --torch-index cu128   # pick the PyTorch build yourself (cu118, cu126, cu128, cu130, xpu, cpu ...)
    python run.py --no-venv          # use the current Python instead of .venv
    python run.py --no-fast          # skip llama-cpp-python (fast quantized chat on CPU)
    python run.py --lite             # skip the AI toolkit extras (requirements-ai.txt / -gpu.txt)
    python run.py --join http://ADDRESS:7861 --secret SECRET
                                     # don't start the Studio: set up, then lend this PC (CPU or GPU)
                                     # to a cluster another PC started (Training → Cluster). Other
                                     # worker options pass through, e.g. --name, --device, --data-mb

What it does:
  1. Creates a virtual environment in .venv (first run only).
  2. Installs requirements.txt into it. If this PC has an NVIDIA GPU (found with nvidia-smi),
     it installs the CUDA build of PyTorch so training and chat run on the GPU; otherwise the
     much smaller CPU build. A CPU build installed earlier is swapped for the CUDA build once a
     GPU is found. Macs use the regular build, which runs on the Apple GPU (MPS). Packages are
     only reinstalled when requirements.txt or the PyTorch build changes.
  3. Tries once to install llama-cpp-python (prebuilt CPU wheel), which makes chat with
     fine-tuned models several times faster through 4-bit GGUF copies. If it can't be
     installed, everything still works, just slower.
  4. Installs the AI toolkit (requirements-ai.txt: TRL, lm-eval, sentence-transformers, FAISS,
     dedup tools, GGUF, TensorBoard) and, with an NVIDIA GPU, requirements-gpu.txt
     (bitsandbytes, Liger kernels). Best effort: a package that fails is reported and skipped.
  5. Creates the folders the app writes to (runs/, data/raw/, data/bin/).
  6. Starts the web app (app.py) and opens it in your browser. If the port is busy it
     uses the next free one.
Stop it with Ctrl+C.

This file only uses the Python standard library, so it runs before anything is installed.
"""
import argparse, hashlib, os, shutil, socket, subprocess, sys, threading, time, venv, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQS = os.path.join(ROOT, "requirements.txt")
STAMP = os.path.join(VENV, ".requirements.sha256")
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
TORCH_CPU_INDEX = TORCH_INDEX.format("cpu")
LLAMA_CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
FAST_STAMP = os.path.join(VENV, ".fast-attempted")
AI_REQS = os.path.join(ROOT, "requirements-ai.txt")
GPU_REQS = os.path.join(ROOT, "requirements-gpu.txt")
EXTRAS_STAMP = os.path.join(VENV, ".extras.sha256")


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


def req_lines(path):
    """Requirement lines of a requirements file, without comments and blank lines."""
    with open(path) as f:
        return [l.split("#")[0].strip() for l in f if l.split("#")[0].strip()]


def requirements_met(py, path=REQS):
    """True if every package in a requirements file (default requirements.txt) is installed for
    this Python. Lines whose environment marker doesn't apply here (e.g. Linux only) are skipped."""
    code = ("import sys, importlib.metadata as m, re\n"
            "for line in sys.argv[1:]:\n"
            "    req, _, marker = line.partition(';')\n"
            "    if marker.strip():\n"
            "        try:\n"
            "            from packaging.markers import Marker\n"
            "            if not Marker(marker.strip()).evaluate(): continue\n"
            "        except ImportError: pass\n"
            "    m.version(re.split(r'[\\[<>=!~ ]', req.strip())[0])")
    return subprocess.call([py, "-c", code, *req_lines(path)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def torch_build(py):
    """'cuda', 'cpu' or None (torch not installed) for the PyTorch in this Python."""
    code = ("import torch; v = torch.version; "
            "print('cuda' if v.cuda or getattr(v, 'hip', None) or getattr(v, 'xpu', None) else 'cpu')")
    try:
        out = subprocess.run([py, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def has_module(py, name):
    return subprocess.call([py, "-c", f"import {name}"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def nvidia_gpus():
    """[(name, memory_mb, driver_version)] from nvidia-smi, or [] when there is no NVIDIA GPU/driver."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = p if os.path.exists(p) else None
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    gpus = []
    for line in out.stdout.splitlines() if out.returncode == 0 else []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append((parts[0], int(float(parts[1])), parts[2]))
            except ValueError:
                gpus.append((parts[0], 0, parts[2]))
    return gpus


def cuda_index_for(gpus):
    """PyTorch CUDA builds that fit the driver, best first (tried in order, since each PyTorch
    release only publishes a few CUDA versions). CUDA 13 needs driver 580+; CUDA 12 builds run on
    525+ (Linux) / 528+ (Windows) and 12.8 is the first with RTX 50xx support; older drivers get 11.8."""
    driver = gpus[0][2]
    try:
        major = int(driver.split(".")[0])
    except ValueError:
        major = 999
    if major < (528 if os.name == "nt" else 525):
        return "cu118"
    if major >= 580:
        return "cu130,cu128,cu126"
    return "cu128,cu126"


def choose_torch(a):
    """Which PyTorch build to install: 'cpu', 'cuXXX', another index name, or 'default' (macOS / PyPI)."""
    if a.torch_index:
        return a.torch_index
    if sys.platform == "darwin":
        return "default"  # the macOS wheel includes the Apple GPU (MPS) backend
    if a.cpu:
        return "cpu"
    gpus = nvidia_gpus()
    if gpus:
        name, mem, driver = gpus[0]
        say(f"Found GPU: {name} ({mem / 1024:.1f} GB, driver {driver}). Training and chat will use it.")
        return cuda_index_for(gpus)
    if a.gpu:
        say("--gpu: no NVIDIA driver found (nvidia-smi), installing a CUDA 12 build anyway.")
        return "cu128,cu126"
    return "cpu"


def ensure_packages(py, variant, reinstall=False):
    with open(REQS, "rb") as f:
        want = hashlib.sha256(f.read() + variant.encode()).hexdigest()
    stamp = STAMP if py == venv_python() else None
    if stamp and not reinstall and os.path.exists(stamp):
        with open(stamp) as f:
            if f.read().strip() == want:
                return
    have = torch_build(py)
    gpu_wanted = variant.startswith(("cu", "rocm", "xpu"))
    if not stamp and not reinstall and requirements_met(py) and (not gpu_wanted or have == "cuda"):
        return  # --no-venv and everything in requirements.txt is already there
    say("Installing Python packages (first run takes a few minutes) ...")
    pip(py, "--upgrade", "pip")
    if variant != "default":
        if have == "cpu" and gpu_wanted:
            say(f"Replacing the CPU build of PyTorch with the GPU (CUDA) build, about 2.5 GB ...")
            subprocess.call([py, "-m", "pip", "uninstall", "-y", "torch"], cwd=ROOT)
            have = None
        if have is None or reinstall:
            # PyPI's torch is CPU-only on Windows and a large CUDA build on Linux, so always
            # install from the PyTorch index that matches the hardware.
            done = False
            for v in variant.split(","):
                if pip(py, "torch", "--index-url", TORCH_INDEX.format(v.strip()), "--retries", "2", required=False):
                    done = True
                    break
            if not done:
                if gpu_wanted:
                    say(f"Could not install the {variant} build of PyTorch; trying the CPU build so the app still works.")
                if not pip(py, "torch", "--index-url", TORCH_CPU_INDEX, "--retries", "1", required=False):
                    say("Could not reach the PyTorch index, installing the regular PyTorch build instead.")
    pip(py, "-r", REQS)
    if gpu_wanted:
        check = ("import torch; ok = torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available());"
                 "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); raise SystemExit(0 if ok else 1)")
        r = subprocess.run([py, "-c", check], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            say(f"PyTorch sees the GPU: {r.stdout.strip() or variant}.")
        else:
            say("PyTorch was installed but can't use the GPU. Update the NVIDIA driver "
                "(https://www.nvidia.com/Download/index.aspx), then run this again with --reinstall. "
                "Until then everything runs on the CPU.")
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


def ensure_extras(py, gpu, skip=False, reinstall=False):
    """Best effort: the AI toolkit (requirements-ai.txt) and, on NVIDIA GPUs, requirements-gpu.txt.
    Reinstalled only when those files change; a package that can't be installed is skipped."""
    if skip:
        return
    files = [AI_REQS] + ([GPU_REQS] if gpu else [])
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            digest.update(f.read())
    want = digest.hexdigest()
    stamp = EXTRAS_STAMP if py == venv_python() else None
    if not reinstall:
        if stamp and os.path.exists(stamp):
            with open(stamp) as f:
                if f.read().strip() == want:
                    return
        if not stamp and all(requirements_met(py, p) for p in files):
            return
    say("Installing the AI toolkit (" + " + ".join(os.path.basename(p) for p in files) + ", optional) ...")
    failed = []
    for path in files:
        if pip(py, "-r", path, "--retries", "2", required=False):
            continue
        # One package failing makes pip install nothing, so retry them one at a time.
        for line in req_lines(path):
            if not pip(py, line, "--retries", "1", required=False):
                failed.append(line.split(";")[0].strip())
    if failed:
        say("Skipped (the app works without them): " + ", ".join(failed) +
            ". Try again later with: run.bat --reinstall")
    else:
        say("AI toolkit installed. Check it any time with: python stack_check.py")
    if stamp:
        with open(stamp, "w") as f:
            f.write(want)


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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="install the CUDA PyTorch build even if no GPU is detected")
    g.add_argument("--cpu", action="store_true", help="install the CPU PyTorch build and train on the CPU")
    g.add_argument("--torch-index", help="PyTorch build to install: cu118, cu126, cu128, xpu (Intel Arc), cpu ...")
    ap.add_argument("--no-venv", action="store_true", help="use the current Python instead of .venv")
    ap.add_argument("--no-fast", action="store_true", help="don't install llama-cpp-python")
    ap.add_argument("--lite", action="store_true", help="don't install the AI toolkit extras")
    ap.add_argument("--join", metavar="URL", help="join a training cluster as a worker instead of starting the Studio")
    ap.add_argument("--secret", help="the cluster's secret (with --join)")
    a, extra = ap.parse_known_args()
    if extra and not a.join:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    if a.join and not a.secret:
        ap.error("--join needs --secret (shown on the coordinator's Training → Cluster page)")

    if sys.version_info < (3, 9):
        sys.exit(f"[run] Python 3.9 or newer is needed (this is {sys.version.split()[0]}).")
    os.chdir(ROOT)

    py = sys.executable if a.no_venv else ensure_venv()
    variant = choose_torch(a)
    ensure_packages(py, variant, reinstall=a.reinstall)
    ensure_fast(py, skip=a.no_fast)
    ensure_extras(py, gpu=variant.startswith("cu"), skip=a.lite, reinstall=a.reinstall)
    ensure_dirs()

    if a.cpu:
        os.environ["TEXTLLM_DEVICE"] = "cpu"  # the app and every training job it starts stay on the CPU
    if a.join:
        say(f"Joining the training cluster at {a.join}  (Ctrl+C to leave)")
        proc = subprocess.Popen([py, os.path.join(ROOT, "cluster.py"), "worker", "--join", a.join, *extra], cwd=ROOT,
                                env={**os.environ, "TEXTLLM_CLUSTER_SECRET": a.secret})
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:  # the worker got Ctrl+C too and says goodbye to the coordinator
            try:
                sys.exit(proc.wait(timeout=30))
            except subprocess.TimeoutExpired:
                proc.kill()
                sys.exit(1)
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
