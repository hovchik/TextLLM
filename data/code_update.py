#!/usr/bin/env python3
"""Scheduled code updates for the copilot: collect new permissively licensed code, then (optionally)
keep training a copilot model on just the new code.

One run:
  1. GitHub: adds up to N new public repositories (your languages, topics and minimum stars), checks the
     ones collected before plus your watchlist for new commits and adds only their new or changed
     files. Same licence rule as data/build_code.py: permissive licences only, with a LICENSE file.
  2. Your project folders (if switched on): read again; files that changed are added.
  3. Every file that is new in this run is also copied to runs/_code_auto/pending/.
  4. Training (if a copilot model is chosen): once enough new files are waiting, the model trains on
     them (a LoRA copilot continues its adapter, a from-scratch copilot continues from its checkpoint,
     mixed with some of the code it learned before). Then the pending files are removed.

Settings come from the Studio (Training → Copilot → Automatic updates), saved in
runs/_code_auto/settings.json; GitHub settings and the token from data/code/sources.json. Nothing
secret goes on the command line, so the OS scheduler runs simply `python data/code_update.py`.

  python data/code_update.py                         # one run now
  python data/code_update.py --install-schedule 24   # every 24 h, also when the Studio is closed
  python data/code_update.py --uninstall-schedule
"""
import argparse, glob, hashlib, json, os, shutil, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATE = os.path.join(ROOT, "runs", "_code_auto")
SETTINGS = os.path.join(STATE, "settings.json")
STATE_P = os.path.join(STATE, "state.json")
PENDING = os.path.join(STATE, "pending")
LOG = os.path.join(STATE, "log.txt")
LOCAL_SEEN = os.path.join(STATE, "local.hashes")   # project files already queued (or trained on before)
CODE = os.path.join(ROOT, "data", "code")
CODE_CFG = os.path.join(CODE, "sources.json")
JOB_LOCK = os.path.join(ROOT, "runs", "_job.lock")
TASK_NAME = "TextLLM code updates"
EOT = "<|endoftext|>"

DEFAULTS = {
    "enabled": False,       # the Studio's own scheduler runs it (while the Studio is open)
    "every_hours": 24,
    "github_new": 10,       # new repositories per run
    "refresh": True,        # look for new commits in repositories collected before
    "refresh_limit": 100,   # ...at most this many per run (GitHub allows 60 API calls an hour without a token)
    "topics": "",           # GitHub topics, e.g. "cli,django" (empty = any)
    "local": False,         # read the project folders again too
    "train_model": "",      # runs/<name> of a copilot model to keep training ("" = only collect)
    "train_steps": 100,
    "min_files": 50,        # train once this many new files are waiting
    "mix": 1.0,             # from-scratch copilots: earlier code tokens per new token
}


def say(*a):
    print(*a, flush=True)


def _load(p, default):
    try:
        return {**default, **json.load(open(p))}
    except (OSError, ValueError):
        return dict(default)


def _save(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


def load_settings():
    return _load(SETTINGS, DEFAULTS)


def save_settings(s):
    _save(SETTINGS, {k: s[k] for k in DEFAULTS if k in s})


def load_state():
    return _load(STATE_P, {"last_start": 0, "last_end": 0, "last_result": "", "pending_files": 0,
                           "pending_shard": 0, "pending_bytes": 0, "trained_files": 0, "runs": 0})


def update_state(**kw):
    st = load_state()
    st.update(kw)
    _save(STATE_P, st)
    return st


def code_cfg():
    from data.build_code import PERMISSIVE

    c = _load(CODE_CFG, {"local_dirs": [], "github_list": "", "min_stars": 50, "max_repo_mb": 40, "github_token": "",
                         "langs": "python,javascript,typescript", "licenses": PERMISSIVE, "max_file_kb": 256,
                         "retries": 30})
    c["licenses"] = [x for x in (x.lower() for x in c.get("licenses") or []) if x in PERMISSIVE] or PERMISSIVE
    return c


# ---------------------------------------------------------------- locks
def _alive(pid):
    from data.crawl import _alive as alive

    return alive(pid)


def _read(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


class job_lock:
    """Hold runs/_job.lock for the whole run, so the Studio doesn't start a data or training job on the same
    files meanwhile. When the Studio started this run, the lock is already ours. Raises Busy when another
    job holds it."""

    class Busy(Exception):
        pass

    def __enter__(self):
        info = _read(JOB_LOCK)
        if info and _alive(info.get("pid")) and info["pid"] not in (os.getpid(), os.getppid()):
            raise self.Busy(f"{info.get('kind', 'a job')}: {info.get('name', '')}")
        self.mine = not (info and info.get("pid") in (os.getpid(), os.getppid()))
        if self.mine:
            os.makedirs(os.path.dirname(JOB_LOCK), exist_ok=True)
            _save(JOB_LOCK, {"pid": os.getpid(), "kind": "data", "name": "code updates (scheduled)",
                             "started": time.time()})
        return self

    def __exit__(self, *exc):
        if self.mine:
            try:
                os.remove(JOB_LOCK)
            except OSError:
                pass


# ---------------------------------------------------------------- what is new in this run
def _docs(path, start=0):
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read().decode("utf-8", "replace")
    return [d.strip() for d in text.split(EOT) if d.strip()]


def _hash(doc):
    return hashlib.sha1(doc.encode("utf-8", "replace")).hexdigest()[:20]


def local_hashes():
    return {_hash(d) for p in glob.glob(os.path.join(CODE, "local", "*", "shard_*.txt")) for d in _docs(p)}


def github_mark():
    st = _read(os.path.join(CODE, "github", "progress.json")) or {}
    return st.get("shard", 0), st.get("shard_bytes", 0)


def queued_mark():
    """Where the last run's GitHub files were queued up to. It moves only after they are safely in pending/,
    so a run that fails halfway is picked up again by the next one."""
    now = github_mark()
    m = load_state().get("gh_queued")
    if m is None:  # first run: start from here, and remember it before collecting anything
        update_state(gh_queued=list(now))
        return now
    m = tuple(m)
    return m if m <= now else (0, 0)  # the GitHub code was deleted and collected again


def load_local_seen():
    try:
        return set(open(LOCAL_SEEN).read().split())
    except OSError:
        return None


def github_new_docs(mark):
    """GitHub shards only grow: everything after the checkpoint saved before this run is new."""
    idx, size = mark
    out = []
    for p in sorted(glob.glob(os.path.join(CODE, "github", "shard_*.txt"))):
        n = int(os.path.basename(p)[6:11])
        if n >= idx:
            out += _docs(p, size if n == idx else 0)
    return out


def add_pending(docs):
    from data.build_corpus import ShardWriter

    if not docs:
        return load_state()["pending_files"]
    st = load_state()
    if not glob.glob(os.path.join(PENDING, "shard_*.txt")):
        st.update(pending_shard=0, pending_bytes=0, pending_files=0)
    w = ShardWriter(PENDING, st["pending_shard"], st["pending_bytes"])
    for d in docs:
        w.write(d)
    idx, size = w.checkpoint()
    w.close()
    return update_state(pending_shard=idx, pending_bytes=size, pending_files=st["pending_files"] + len(docs))["pending_files"]


def clear_pending():
    shutil.rmtree(PENDING, ignore_errors=True)
    update_state(pending_shard=0, pending_bytes=0, pending_files=0)


# ---------------------------------------------------------------- training on the new code
def last_step(run):
    p = os.path.join(run, "metrics.jsonl")
    steps = [json.loads(l).get("step", 0) for l in open(p) if l.strip()] if os.path.exists(p) else []
    return max(steps) if steps else 0


def lora_cmd(run, meta, steps, empty_dir):
    bs, acc = 2, 8
    return [sys.executable, "finetune/finetune_lora.py", "--base", meta["base"], "--raw", empty_dir,
            "--examples", "0", "--code", PENDING, "--code-examples", str(steps * bs * acc),
            "--fim-rate", "0.5",
            "--out", run, "--init-adapter", os.path.join(run, "adapter"), "--seq", "512",
            "--batch", str(bs), "--grad-accum", str(acc), "--max-steps", str(steps), "--lr", "1e-4",
            "--eval-every", str(max(1, steps // 5)), "--save-every", str(steps),
            "--step-offset", str(last_step(run)), "--private-texts"]


def same_tokenizer(tok_p):
    """data/bin-code is encoded with tokenizer/bpe-code.json; its token ids mean the same to this model only
    when the model's own tokenizer copy is that same tokenizer (a rebuilt or imported one may differ)."""
    shared = os.path.join(ROOT, "tokenizer", "bpe-code.json")
    try:
        return open(tok_p, "rb").read() == open(shared, "rb").read()
    except OSError:
        return False


def scratch_cmd(run, meta, steps, mix, tmp):
    """Encode the new code with the model's own tokenizer (with fill-in-the-middle), mix in some of
    data/bin-code (what it learned before) and continue from the checkpoint."""
    import numpy as np
    from core import safeload
    from tokenizer.train_tokenizer import encode

    ck = safeload.load(os.path.join(run, "ckpt.pt"))
    cfg, step = ck["config"], ck["step"]
    del ck
    bindir = os.path.join(tmp, "bin")
    tok_p = os.path.join(run, meta.get("tok", "tokenizer.json"))
    encode(PENDING, tok_p, bindir, 0.0, fim_rate=0.5,
           seed=int(time.time()))
    new = np.fromfile(os.path.join(bindir, "train.bin"), dtype=np.uint16)
    block = cfg["block_size"]
    if len(new) < block + 2:
        raise SystemExit(f"the new code is too short to train on ({len(new)} tokens)")
    parts = [new]
    old_p = os.path.join(ROOT, "data", "bin-code", "train.bin")
    if mix > 0 and os.path.exists(old_p) and not same_tokenizer(tok_p):
        say("not mixing in data/bin-code: it was encoded with a different code tokenizer than this model's")
    elif mix > 0 and os.path.exists(old_p):
        old = np.memmap(old_p, dtype=np.uint16, mode="r")
        n = min(len(old) - 1, int(len(new) * mix))
        if n > block:
            s = np.random.default_rng().integers(0, len(old) - n)
            parts.append(np.array(old[s:s + n]))
    np.concatenate(parts).tofile(os.path.join(bindir, "train.bin"))
    new.tofile(os.path.join(bindir, "val.bin"))  # "val" = how well it now knows the new code
    say(f"{len(new):,} new tokens (+{sum(len(p) for p in parts[1:]):,} earlier ones) → {steps} steps from step {step}")
    return [sys.executable, "train.py", "--data", bindir, "--out", run, "--resume", "--preset", meta.get("preset", "cpu"),
            "--vocab", str(cfg["vocab_size"]), "--max-steps", str(step + steps), "--lr", "3e-4", "--min-lr", "3e-4",
            "--warmup", "1", "--eval-every", str(max(1, steps // 5)), "--eval-iters", "5"]


def train(run, steps, mix):
    from graceful import run_child

    meta = json.load(open(os.path.join(run, "meta.json")))
    if meta.get("task") != "code":
        raise SystemExit(f"{os.path.basename(run)} is not a copilot model; choose one trained in Training → Copilot")
    tmp = tempfile.mkdtemp(prefix="code_update_")
    try:
        if meta.get("type") == "scratch":
            cmd = scratch_cmd(run, meta, steps, mix, tmp)
        else:
            empty = os.path.join(tmp, "no-text")
            os.makedirs(empty)
            cmd = lora_cmd(run, meta, steps, empty)
        rc = run_child(cmd, ROOT)
        if rc != 0:
            raise SystemExit(f"training failed (exit {rc}); the new code stays queued for the next run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- one run
def run_once(s=None, token=None):
    from coding import langs as L
    from data import build_code as B

    s = s or load_settings()
    c = code_cfg()
    token = token if token is not None else (os.environ.get("GITHUB_TOKEN") or c.get("github_token") or "")
    update_state(last_start=time.time(), last_result="running")
    langs = L.parse_langs(c["langs"])
    watch = [x.strip().removeprefix("https://github.com/").strip("/") for x in
             c.get("github_list", "").replace(",", "\n").split() if x.strip()]
    topics = B.parse_topics(s.get("topics", ""))
    new_docs = []
    os.makedirs(CODE, exist_ok=True)
    if s["github_new"] > 0 or s["refresh"] or watch:
        mark = queued_mark()
        say(f"== GitHub: {s['github_new']} new repositories"
            + (f" (topics: {', '.join(topics)})" if topics else "")
            + (", checking for new commits" if s["refresh"] else "") + (f", watchlist {len(watch)}" if watch else ""))
        B.github(CODE, 0, watch, langs, c["licenses"], max(0, int(c["min_stars"])), int(c["max_file_kb"]),
                 float(c["max_repo_mb"]), int(c["retries"]), token, new=max(0, int(s["github_new"])), topics=topics,
                 refresh=bool(s["refresh"]), refresh_limit=max(0, int(s["refresh_limit"])))
        gh = github_new_docs(mark)
        say(f"github: {len(gh)} new files in this run")
        new_docs += gh
    seen = None
    if s["local"] and c["local_dirs"]:
        seen = load_local_seen()
        if seen is None:  # first time: what is there now was part of the normal training
            seen = local_hashes()
        say(f"== your projects: reading {len(c['local_dirs'])} folder(s) again")
        B.local(c["local_dirs"], CODE, langs, int(c["max_file_kb"]))
        now = [d for p in glob.glob(os.path.join(CODE, "local", "*", "shard_*.txt")) for d in _docs(p)]
        loc = [d for d in now if _hash(d) not in seen]
        seen |= {_hash(d) for d in now}
        say(f"local: {len(loc)} new or changed files")
        new_docs += loc
    pending = add_pending(new_docs)
    # only now that the files are in pending/ do the cursors move on
    if s["github_new"] > 0 or s["refresh"] or watch:
        update_state(gh_queued=list(github_mark()))
    if seen is not None:
        tmp = LOCAL_SEEN + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(sorted(seen)))
        os.replace(tmp, LOCAL_SEEN)
    files = lambda n: f"{n} new file" + ("" if n == 1 else "s")
    result = f"{files(len(new_docs))}; {pending} waiting for training"
    run = s.get("train_model")
    if run:
        path = os.path.join(ROOT, "runs", run)
        if not os.path.isdir(path):
            say(f"training: {run} no longer exists; only collecting")
        elif pending < max(1, int(s["min_files"])):
            say(f"training: {pending} new files waiting, training starts at {s['min_files']}")
        else:
            say(f"== training {run} on {pending} new files ({s['train_steps']} steps)")
            train(path, max(1, int(s["train_steps"])), float(s.get("mix", 1.0)))
            st = load_state()
            update_state(trained_files=st["trained_files"] + pending, last_train=time.time())
            clear_pending()
            result = f"{files(len(new_docs))}; {run} trained on {pending} files"
    return result


def schedule_status():
    from core import schedule

    return schedule.status(TASK_NAME)


def install_schedule(every_hours):
    from core import schedule

    return schedule.install(TASK_NAME, [sys.executable, os.path.join(ROOT, "data", "code_update.py")], LOG,
                            every_hours, ROOT, "the code updates")


def uninstall_schedule():
    from core import schedule

    return schedule.remove(TASK_NAME)


def main():
    import graceful

    graceful.install()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install-schedule", type=float, metavar="HOURS",
                    help="register with Task Scheduler (Windows) or cron, so updates run when the Studio is closed")
    ap.add_argument("--uninstall-schedule", action="store_true")
    a = ap.parse_args()
    if a.install_schedule:
        return say(install_schedule(a.install_schedule))
    if a.uninstall_schedule:
        return say(uninstall_schedule())
    say(f"\n=== code updates {time.strftime('%Y-%m-%d %H:%M')}")
    try:
        with job_lock():
            result = run_once()
    except job_lock.Busy as e:
        say(f"the Studio is busy ({e}); trying again next time")
        update_state(last_result=f"skipped: the Studio was busy ({e})")
        return
    except KeyboardInterrupt:
        update_state(last_end=time.time(), last_result="stopped")
        sys.exit("\nStopped. New files collected so far are kept and stay queued for training.")
    except SystemExit as e:
        update_state(last_end=time.time(), last_result=f"failed: {e}")
        raise
    except Exception as e:
        update_state(last_end=time.time(), last_result=f"failed: {e}")
        raise
    st = load_state()
    update_state(last_end=time.time(), last_result=result, runs=st["runs"] + 1)
    say(f"✅ {result}")


if __name__ == "__main__":
    main()
