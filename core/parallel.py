"""Train with several processes on one PC (data parallel), and pick how many CPU threads each uses.

One process with all cores is what train.py always did. PyTorch spreads each matrix multiply over
the cores, but the many small operations of a small model (norms, softmax, the optimizer step) run
on one core or don't split well, so the cores sit idle part of the time. Several processes, each
with its own slice of the batch and a share of the cores, keep them busier; after every step they
average their gradients (torch.distributed, "gloo" on the CPU), so the result is the same model
as one process training on the whole batch.

    python train.py --procs 2 --threads 0     # 2 processes, all physical cores split between them

The same launcher runs one process per GPU when the PC has several NVIDIA GPUs ("nccl" on Linux,
"gloo" on Windows, where NCCL doesn't exist).
"""
import json, os, re, shutil, socket, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PATH = os.path.join(ROOT, "runs", "_parallel.json")  # speed test results, per model size

ENV_RANK = "RANK"
ENV_SPLIT = "TEXTLLM_SPLIT_BATCH"  # set by launch(): --batch is the total over all processes


def physical_cores():
    from writing import hf

    return hf.physical_cores()


def total_threads(threads=0):
    """CPU threads the whole run may use: --threads, else TEXTLLM_THREADS, else the physical cores."""
    return int(threads or 0) or int(os.environ.get("TEXTLLM_THREADS", 0) or 0) or physical_cores()


def auto_procs(device_kind, gpus=0, cores=None, preset=None):
    """How many processes 'auto' starts. One per GPU when there are several. On the CPU it is
    what the speed test (bench() below, Studio: Training → Find the fastest) measured as fastest
    for this model size on this PC; one process until it has been run, because on many CPUs one
    process already keeps every core busy (a 4-core test machine ran 1, 2 and 4 processes within
    5% of each other)."""
    if device_kind == "cuda":
        return max(1, gpus)
    if device_kind != "cpu":
        return 1  # Apple / Intel GPUs: one device, one process
    best = load_bench().get(preset or "", {}).get("best")
    return int(best) if best else 1


def plan(procs, threads, device_kind, gpus=0, preset=None):
    """(processes, threads per process). procs 0 = auto."""
    n = int(procs or 0) or auto_procs(device_kind, gpus, preset=preset)
    if device_kind == "cuda":
        n = max(1, min(n, max(1, gpus)))
    elif device_kind != "cpu":
        n = 1
    total = total_threads(threads)
    n = max(1, min(n, total)) if device_kind == "cpu" else n
    return n, max(1, total // n)


def split_batch(batch, procs):
    """Each process's micro-batch, so the total batch stays what was asked for (rounded up)."""
    return max(1, -(-int(batch) // max(1, procs)))


def backend(device_kind):
    if device_kind == "cuda" and os.name != "nt":
        return "nccl"
    return "gloo"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def child_env(rank, world, port, threads):
    env = {**os.environ, ENV_RANK: str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
           "LOCAL_WORLD_SIZE": str(world), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
           ENV_SPLIT: "1", "TEXTLLM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    if os.name == "nt":
        env["USE_LIBUV"] = "0"  # Windows PyTorch builds have no libuv for the rendezvous store
    return env


def launch(script, argv, procs, threads, grace=120):
    """Run `script argv` `procs` times as one data-parallel job and wait. Returns the exit code.

    If one process fails the others are stopped (they would wait forever for its gradients).
    Ctrl-C / Studio Stop reaches all of them (same process group); this waits for them to exit."""
    import graceful

    port = free_port()
    kids = [subprocess.Popen([sys.executable, script, *argv], env=child_env(r, procs, port, threads))
            for r in range(procs)]
    rc = 0
    try:
        while True:
            codes = [k.poll() for k in kids]
            bad = [c for c in codes if c not in (None, 0)]
            if bad:
                rc = bad[0]
                time.sleep(3)  # let the others notice and print their own error first
                for k in kids:
                    if k.poll() is None:
                        graceful.kill(k)
                break
            if all(c == 0 for c in codes):
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        end = time.time() + grace
        while any(k.poll() is None for k in kids) and time.time() < end:
            try:
                time.sleep(0.2)
            except KeyboardInterrupt:
                pass
        for k in kids:
            if k.poll() is None:
                graceful.kill(k)
        raise
    for k in kids:
        k.wait()
    return rc


# ---------------------------------------------------------------- speed test
def load_bench(path=None):
    try:
        d = json.load(open(path or BENCH_PATH))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _ms_per_step(log):
    """Mean ms/step of the logged steps after the first (which includes start-up and warm-up)."""
    ms = [int(m) for m in re.findall(r"^step \d+ .*\| (\d+) ms/step", log, re.M)]
    ms = ms[1:] or ms
    return sum(ms) / len(ms) if ms else None


def bench(preset="cpu", vocab=8192, batch=16, accum=1, steps=30, candidates=None, threads=0,
          device="cpu", save=True, path=None, log=print):
    """Train `steps` steps on random tokens with 1, 2, 4… processes and keep the fastest.

    Returns {"best": procs, "results": {procs: ms_per_step}, ...}. A count only wins if it is at
    least 5% faster than one process (smaller differences are noise)."""
    total = total_threads(threads)
    cands = sorted({c for c in (candidates or (1, 2, 4)) if 1 <= c <= total} | {1})
    tmp = tempfile.mkdtemp(prefix="textllm-bench-")
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        data = os.path.join(tmp, "bin")
        os.makedirs(data)
        rng.integers(0, vocab, 2_000_000, dtype=np.uint16).tofile(os.path.join(data, "train.bin"))
        rng.integers(0, vocab, 50_000, dtype=np.uint16).tofile(os.path.join(data, "val.bin"))
        results = {}
        for n in cands:
            log(f"speed test: {n} process(es), {max(1, total // n)} threads each …")
            cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--data", data, "--out", os.path.join(tmp, f"o{n}"),
                   "--preset", preset, "--vocab", str(vocab), "--batch", str(batch), "--grad-accum", str(accum),
                   "--max-steps", str(steps), "--eval-every", str(steps * 10), "--eval-iters", "1",
                   "--warmup", "5", "--device", device, "--procs", str(n), "--threads", str(total)]
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            env.pop(ENV_RANK, None)
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
            ms = _ms_per_step(p.stdout)
            if p.returncode != 0 or ms is None:
                log(f"  failed: {(p.stdout + p.stderr).strip().splitlines()[-1:] or ['no output']}")
                continue
            results[n] = round(ms)
            log(f"  {ms:.0f} ms/step")
        if not results:
            raise RuntimeError("the speed test could not train at all; see the messages above")
        one = results.get(1)
        best = min(results, key=results.get)
        if one and results[best] > one * 0.95:
            best = 1
        out = {"best": best, "results": results, "threads": total, "batch": batch, "accum": accum,
               "device": device, "time": time.time()}
        if save:
            allb = load_bench(path)
            allb[preset] = out
            os.makedirs(os.path.dirname(path or BENCH_PATH), exist_ok=True)
            json.dump(allb, open(path or BENCH_PATH, "w"), indent=1)
        if best == 1:
            log(f"fastest for {preset}: 1 process (more processes were not at least 5% faster here)")
        else:
            log(f"fastest for {preset}: {best} processes, {one / results[best]:.2f}x the speed of 1" if one
                else f"fastest for {preset}: {best} processes")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, ROOT)
    import graceful

    graceful.install()
    ap = argparse.ArgumentParser(description="Find how many training processes are fastest on this PC.")
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--candidates", default="1,2,4")
    a = ap.parse_args()
    bench(a.preset, a.vocab, a.batch, a.grad_accum, a.steps,
          [int(c) for c in a.candidates.split(",") if c.strip().isdigit()], a.threads,
          log=lambda m: print(m, flush=True))
