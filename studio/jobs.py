"""One background job at a time (data prep, training, learning, feedback training)."""
import glob, json, math, os, signal, subprocess, sys, threading, time

from chat_engine import ROOT, RUNS

PY = sys.executable
os.makedirs(RUNS, exist_ok=True)
_lock = threading.Lock()
JOB = {"proc": None, "name": None, "kind": None, "log": None, "run": None, "started": None, "cleanup": []}
HISTORY_PATH = os.path.join(RUNS, "jobs.jsonl")


def running():
    return JOB["proc"] is not None and JOB["proc"].poll() is None


def launch(kind, name, cmd, log_path, run=None, cleanup=()):
    with _lock:
        if running():
            raise RuntimeError(f"A job is already running ({JOB['kind']}: {JOB['name']}). Stop it first.")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logf = open(log_path, "a")
        logf.write(f"\n$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        JOB.update(proc=proc, name=name, kind=kind, log=log_path, run=run, started=time.time(),
                   cleanup=list(cleanup))
        threading.Thread(target=_watch, args=(proc,), daemon=True).start()
    return status()


def _watch(proc):
    rc = proc.wait()
    for f in JOB["cleanup"]:
        try:
            os.remove(f)
        except OSError:
            pass
    with open(HISTORY_PATH, "a") as h:
        h.write(json.dumps({"kind": JOB["kind"], "name": JOB["name"], "run": JOB["run"], "started": JOB["started"],
                            "ended": time.time(), "exit": rc}) + "\n")


def stop():
    p = JOB["proc"]
    if not running():
        return status()
    p.send_signal(signal.SIGINT)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
    return status()


def tail(path, n=200):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 60000))
        lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
    return "\n".join(lines[-n:])


def status():
    p = JOB["proc"]
    return {
        "running": running(),
        "kind": JOB["kind"], "name": JOB["name"], "run": JOB["run"], "started": JOB["started"],
        "exit": None if p is None or p.poll() is None else p.returncode,
        "log": tail(JOB["log"], 60),
    }


def history(n=50):
    if not os.path.exists(HISTORY_PATH):
        return []
    return [json.loads(l) for l in open(HISTORY_PATH) if l.strip()][-n:]


# ---------------------------------------------------------------- data & runs
def data_status():
    raw = glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True)
    custom = [f for f in raw if os.sep + "custom" + os.sep in f]
    tb = os.path.join(ROOT, "data/bin/train.bin")
    tokp = os.path.join(ROOT, "tokenizer/bpe.json")
    vocab = None
    if os.path.exists(tokp):
        try:
            vocab = len(json.load(open(tokp))["model"]["vocab"])
        except Exception:
            pass
    return {
        "raw_files": len(raw), "raw_mb": round(sum(os.path.getsize(f) for f in raw) / 1e6, 1),
        "custom_files": len(custom),
        "tokenizer": os.path.exists(tokp), "vocab": vocab,
        "train_tokens": os.path.getsize(tb) // 2 if os.path.exists(tb) else 0,
    }


LEVELS = [(6.5, 1, "Random letters"), (5.0, 2, "Real words"), (4.0, 3, "Short phrases"),
          (3.4, 4, "Grammatical sentences"), (3.0, 5, "Fluent paragraphs")]


def level(val_loss, scratch):
    if val_loss is None:
        return None
    if not scratch:
        return {"n": None, "label": "Fine-tuned open model"}
    for th, n, lab in LEVELS:
        if val_loss > th:
            return {"n": n, "label": lab}
    return {"n": 6, "label": "Stays on topic"}


def read_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def run_summary(run_dir):
    name = os.path.basename(run_dir)
    meta_p = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    rows = read_metrics(run_dir)
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    fb = [r for r in rows if r["kind"] == "feedback"]
    scratch = meta.get("type", "scratch") == "scratch"
    step = max((r["step"] for r in rows), default=0)
    total = rows[-1]["max_steps"] if rows else 0
    is_running = running() and JOB["run"] == name
    eta = None
    if is_running and len(tr) >= 2:
        recent = tr[-20:]
        rate = (recent[-1]["step"] - recent[0]["step"]) / max(1e-6, recent[-1]["time"] - recent[0]["time"])
        eta = (total - step) / rate if rate > 0 else None
    last_val = va[-1]["loss"] if va else None
    ready = os.path.exists(os.path.join(run_dir, "ckpt.pt")) if scratch else os.path.isdir(os.path.join(run_dir, "adapter"))
    times = [r["time"] for r in rows]
    return {
        "name": name, "type": "scratch" if scratch else "lora",
        "preset": meta.get("preset"), "base": meta.get("base"),
        "ready": ready, "running": is_running,
        "step": step, "max_steps": total, "progress": (step / total) if total else 0, "eta_sec": eta,
        "train_loss": tr[-1]["loss"] if tr else None,
        "val_loss": last_val, "best_val": min((r["loss"] for r in va), default=None),
        "perplexity": math.exp(min(last_val, 20)) if last_val is not None else None,
        "tokens_per_sec": tr[-1].get("tokens_per_sec") if tr else None,
        "lr": tr[-1].get("lr") if tr else None,
        "level": level(last_val, scratch),
        "feedback_steps": len(fb),
        # active time only: gaps over 15 minutes between log lines are pauses, not training
        "train_seconds": sum(min(b - a, 900) for a, b in zip(sorted(times), sorted(times)[1:])),
        "updated": max(times) if times else os.path.getmtime(run_dir),
    }


def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if os.path.isdir(d) and not os.path.basename(d).startswith("_") and (
                os.path.exists(os.path.join(d, "meta.json")) or os.path.exists(os.path.join(d, "metrics.jsonl"))):
            out.append(run_summary(d))
    return sorted(out, key=lambda r: -r["updated"])
