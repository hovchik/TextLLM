"""One background job at a time (data prep, training, learning, feedback training)."""
import glob, json, math, os, subprocess, sys, threading, time

import graceful
from chat_engine import ROOT, RUNS

PY = sys.executable
os.makedirs(RUNS, exist_ok=True)
_lock = threading.Lock()
JOB = {"proc": None, "name": None, "kind": None, "log": None, "run": None, "started": None, "cleanup": []}
HISTORY_PATH = os.path.join(RUNS, "jobs.jsonl")
LOCK_PATH = os.path.join(RUNS, "_job.lock")   # pid of the running job: data/crawl.py checks it


def running():
    return JOB["proc"] is not None and JOB["proc"].poll() is None


def external_job():
    """A job started outside the Studio (the scheduled crawler training a model), from runs/_job.lock."""
    try:
        info = json.load(open(LOCK_PATH))
    except (OSError, ValueError):
        return None
    if JOB["proc"] is not None and info.get("pid") == JOB["proc"].pid:
        return None  # our own
    from data.crawl import _alive

    return info if _alive(info.get("pid")) else None


def launch(kind, name, cmd, log_path, run=None, cleanup=(), env=None):
    with _lock:
        if running():
            raise RuntimeError(f"A job is already running ({JOB['kind']}: {JOB['name']}). Stop it first.")
        ext = external_job()
        if ext:
            raise RuntimeError(f"The scheduled news crawler is training ({ext.get('name', 'crawl')}, "
                               f"since {time.strftime('%H:%M', time.localtime(ext.get('started', 0)))}). "
                               "Wait for it to finish.")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logf = open(log_path, "a")
        logf.write(f"\n$ {' '.join(cmd)}\n")
        logf.flush()
        # own process group, so Stop reaches the trainers a job starts, not just the job (graceful.py)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, **graceful.popen_kwargs())
        with open(LOCK_PATH, "w") as lk:
            lk.write(json.dumps({"pid": proc.pid, "kind": kind, "name": name, "started": time.time()}))
        JOB.update(proc=proc, name=name, kind=kind, log=log_path, run=run, started=time.time(),
                   cleanup=list(cleanup))
        threading.Thread(target=_watch, args=(proc,), daemon=True).start()
    return status()


def _watch(proc):
    rc = proc.wait()
    _raw_scan["v"] = None  # a data job may have added or removed shards
    try:
        if json.load(open(LOCK_PATH)).get("pid") == proc.pid:
            os.remove(LOCK_PATH)
    except (OSError, ValueError):
        pass
    for f in JOB["cleanup"]:
        try:
            os.remove(f)
        except OSError:
            pass
    with open(HISTORY_PATH, "a") as h:
        h.write(json.dumps({"kind": JOB["kind"], "name": JOB["name"], "run": JOB["run"], "started": JOB["started"],
                            "ended": time.time(), "exit": rc}) + "\n")


# Longer than graceful.run_child's 120 s, so a job that waits for its trainer to save (learn.py)
# gets to finish, and its cleanup (deleting the extracted texts) runs before anything is killed.
STOP_GRACE = 150


def stop(grace=STOP_GRACE, wait=False):
    """Stop the running job and everything it started. Works on Windows too (graceful.py).

    Returns at once; a background thread force-kills the job if it is still running after `grace`
    seconds. wait=True blocks until then (used when the Studio itself shuts down)."""
    p = JOB["proc"]
    if not running():
        return status()
    graceful.interrupt(p)

    def reap():
        if not _wait(p, grace):
            graceful.kill(p)
            _wait(p, 10)

    if wait:
        reap()
    else:
        threading.Thread(target=reap, daemon=True).start()
        _wait(p, 2)  # most jobs exit within a moment; then the status below already says so
    return status()


def _wait(p, seconds):
    """True if the process ended within `seconds`."""
    try:
        p.wait(seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


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
        "running": running(), "external": external_job(),
        "kind": JOB["kind"], "name": JOB["name"], "run": JOB["run"], "started": JOB["started"],
        "exit": None if p is None or p.poll() is None else p.returncode,
        "log": tail(JOB["log"], 60),
    }


def history(n=50):
    if not os.path.exists(HISTORY_PATH):
        return []
    return [json.loads(l) for l in open(HISTORY_PATH) if l.strip()][-n:]


# ---------------------------------------------------------------- data & runs
RAW_SCAN_TTL = 15  # seconds; the Studio asks for the status every 3 s
_raw_scan = {"t": 0.0, "v": None}
_vocab = {"key": None, "v": None}


def _scan_raw():
    """(files, MB, custom files) of data/raw. Walking thousands of book shards on every status
    poll was the slowest part of /api/status, so the result is reused for a few seconds."""
    if _raw_scan["v"] is None or time.time() - _raw_scan["t"] > RAW_SCAN_TTL:
        raw = glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True)
        size = 0
        for f in raw:
            try:
                size += os.path.getsize(f)
            except OSError:  # removed while scanning (a crawl or data job is running)
                pass
        custom = sum(1 for f in raw if os.sep + "custom" + os.sep in f)
        _raw_scan.update(t=time.time(), v=(len(raw), round(size / 1e6, 1), custom))
    return _raw_scan["v"]


def _vocab_size(tokp):
    """Vocabulary size of the tokenizer file, re-read only when the file changes (it is MBs of JSON)."""
    try:
        st = os.stat(tokp)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    if _vocab["key"] != key:
        try:
            v = len(json.load(open(tokp, encoding="utf-8"))["model"]["vocab"])
        except Exception:
            v = None
        _vocab.update(key=key, v=v)
    return _vocab["v"]


def data_status():
    n_raw, raw_mb, n_custom = _scan_raw()
    tb = os.path.join(ROOT, "data/bin/train.bin")
    tokp = os.path.join(ROOT, "tokenizer/bpe.json")
    return {
        "raw_files": n_raw, "raw_mb": raw_mb,
        "custom_files": n_custom,
        "tokenizer": os.path.exists(tokp), "vocab": _vocab_size(tokp),
        "train_tokens": os.path.getsize(tb) // 2 if os.path.exists(tb) else 0,
        "instructions": _instruction_stats(),
    }


def _instruction_stats():
    p = os.path.join(ROOT, "data/instructions/stats.json")
    try:
        return json.load(open(p)).get("train", 0)
    except (OSError, ValueError):
        return 0


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
    from chat_engine import gguf_path

    ev_p = os.path.join(run_dir, "eval", "history.jsonl")
    ev = None
    if os.path.exists(ev_p):
        from evaluate import counts

        try:
            ev = [r for r in (json.loads(l) for l in open(ev_p) if l.strip()) if counts(r)][-1]
        except (ValueError, IndexError):
            ev = None
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
        "fast": bool(not scratch and gguf_path(run_dir)),
        "eval_score": ev["score"] if ev else None, "eval_quick": ev.get("quick") if ev else None,
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
