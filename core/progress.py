"""Progress that the Studio can show: per source (or step) and overall, with bytes, percent and ETA.

Used by the downloads and by local steps such as encoding the corpus (board.kind = "work").

A data script asks for a task per source and moves it forward; the board writes everything to the
JSON file named by $TEXTLLM_PROGRESS (the Studio sets it for every job it starts) at most once a
second, and the Studio reads that file on each status poll. Without the variable nothing is written,
so the scripts behave the same on the command line, where tqdm draws the bars and `summary()` prints
an overall line between sources.

    from core import progress
    board = progress.board()
    board.plan([("gutenberg", "Project Gutenberg", 3000, "books"), ("fineweb", "FineWeb-Edu", 3e8, "bytes")])
    t = board.task("gutenberg")            # or board.task(key, label, total, unit) for an unplanned one
    t.start(done=120)                       # resumed: 120 books were already saved
    t.add(1)                                # one more book
    progress.net(len(chunk))                # bytes that came over the network (any source)
    t.finish()

Nothing here imports anything heavy; it is safe to import from any script.
"""
import json, os, threading, time
from collections import deque

ENV = "TEXTLLM_PROGRESS"
WINDOW = 60.0          # seconds of history used for the speed, so ETA follows the current pace
WRITE_EVERY = 1.0


def fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def fmt_eta(sec):
    if sec is None:
        return "—"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60:02d}s"
    if sec < 86400:
        return f"{sec // 3600}h {sec % 3600 // 60:02d}m"
    return f"{sec // 86400}d {sec % 86400 // 3600}h"


def fmt_amount(n, unit):
    return fmt_bytes(n) if unit == "bytes" else f"{int(n):,} {unit}"


class _Rate:
    """Speed over the last WINDOW seconds."""

    def __init__(self):
        self.s = deque()

    def add(self, t, v):
        self.s.append((t, v))
        while len(self.s) > 2 and t - self.s[0][0] > WINDOW:
            self.s.popleft()

    def get(self):
        if len(self.s) < 2:
            return 0.0
        (t0, v0), (t1, v1) = self.s[0], self.s[-1]
        return (v1 - v0) / (t1 - t0) if t1 > t0 else 0.0


class Task:
    def __init__(self, board, key, label, total, unit):
        self.b, self.key, self.label, self.total, self.unit = board, key, label, float(total or 0), unit
        self.done = 0.0
        self.state = "waiting"          # waiting | running | done | skipped | failed
        self.note = ""
        self.started = None
        self.rate = _Rate()
        self.verb = ""                  # word the Studio shows while running; "" = "Downloading"

    def start(self, done=0, total=None, note=""):
        with self.b.lock:
            if total is not None:
                self.total = float(total)
            self.done, self.state, self.note, self.started = float(done), "running", note, time.time()
            self.rate = _Rate()
            self.rate.add(self.started, self.done)
        self.b.touch(force=True)
        return self

    def add(self, n=1):
        with self.b.lock:
            self.done += n
        self.b.touch()

    def set(self, done=None, total=None, note=None):
        with self.b.lock:
            if done is not None:
                self.done = float(done)
            if total is not None:
                self.total = float(total)
            if note is not None:
                self.note = note
        self.b.touch()

    def finish(self, state="done", note=None):
        with self.b.lock:
            self.state = state
            if note is not None:
                self.note = note
            if state == "done" and self.total and self.done < self.total and not self.note:
                self.note = "the source has no more"
        self.b.touch(force=True)

    def frac(self):
        if self.state in ("done", "skipped"):
            return 1.0
        return min(1.0, self.done / self.total) if self.total > 0 else 0.0

    def info(self, now):
        if self.state == "running":
            self.rate.add(now, self.done)
        speed = self.rate.get() if self.state == "running" else 0.0
        left = max(0.0, self.total - self.done)
        eta = left / speed if self.state == "running" and speed > 0 and self.total else None
        return {"key": self.key, "label": self.label, "unit": self.unit, "state": self.state, "note": self.note,
                "done": self.done, "total": self.total, "pct": round(100 * self.frac(), 1),
                "speed": speed, "eta": eta, "verb": self.verb}


class Board:
    def __init__(self, path=None):
        self.path = path
        self.lock = threading.RLock()
        self.tasks = {}
        self.order = []
        self.started = time.time()
        self.net_bytes = 0
        self.net_rate = _Rate()
        self.title = ""
        self.kind = "download"     # "work" for local steps (encoding, tokenizer): no network figures
        self._last = 0.0
        self._f0 = None            # overall fraction when this run began, for the overall ETA

    # ---- tasks
    def plan(self, items, title=None):
        """items: (key, label, total, unit). Shows every source the run will fetch, in order."""
        for key, label, total, unit in items:
            self.task(key, label, total, unit)
        if title:
            self.title = title
        self.touch(force=True)

    def task(self, key, label=None, total=0, unit="bytes"):
        with self.lock:
            t = self.tasks.get(key)
            if t is None:
                t = self.tasks[key] = Task(self, key, label or key, total, unit)
                self.order.append(key)
            return t

    # ---- network
    def net(self, n):
        with self.lock:
            self.net_bytes += n
        self.touch()

    # ---- output
    def overall(self):
        ts = [self.tasks[k] for k in self.order]
        if not ts:
            return 0.0
        return sum(t.frac() for t in ts) / len(ts)

    def snapshot(self):
        now = time.time()
        with self.lock:
            self.net_rate.add(now, self.net_bytes)
            tasks = [self.tasks[k].info(now) for k in self.order]
            f = self.overall()
            if self._f0 is None or (f < self._f0):
                self._f0 = f
            el = now - self.started
            gained = f - self._f0
            eta = el * (1 - f) / gained if gained > 1e-4 and el > 5 and f < 1 else None
            cur = next((t for t in tasks if t["state"] == "running"), None)
            if cur and cur["eta"] is not None and len([t for t in tasks if t["state"] in ("waiting", "running")]) == 1:
                eta = cur["eta"]       # only one source left: its own ETA is the better guess
            return {"pid": os.getpid(), "title": self.title, "kind": self.kind, "started": self.started, "updated": now,
                    "net_bytes": self.net_bytes, "net_speed": self.net_rate.get(), "tasks": tasks,
                    "overall": {"pct": round(100 * f, 1), "eta": eta,
                                "done": sum(t["state"] in ("done", "skipped") for t in tasks),
                                "count": len(tasks), "current": cur["label"] if cur else None}}

    def touch(self, force=False):
        if not self.path:
            return
        # One writer at a time: download threads and the main thread share the temp file, so the
        # throttle check, snapshot and rename all happen under the board lock.
        with self.lock:
            now = time.time()
            if not force and now - self._last < WRITE_EVERY:
                return
            self._last = now
            try:
                snap = self.snapshot()
                tmp = f"{self.path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(snap, f)
                os.replace(tmp, self.path)
            except OSError:
                pass                 # progress is a nicety; never stop a download over it

    def summary(self):
        """One line for the command line: overall percent, ETA and bytes downloaded."""
        s = self.snapshot()
        o = s["overall"]
        return (f"[overall {o['pct']:.0f}% · {o['done']}/{o['count']} sources done · "
                f"{fmt_bytes(s['net_bytes'])} downloaded"
                + (f" at {fmt_bytes(s['net_speed'])}/s" if s["net_speed"] > 0 else "")
                + (f" · ETA {fmt_eta(o['eta'])}" if o["eta"] is not None else "") + "]")


_BOARD = None


def board():
    """The process-wide board, writing to $TEXTLLM_PROGRESS when the Studio set it."""
    global _BOARD
    if _BOARD is None:
        _BOARD = Board(os.environ.get(ENV) or None)
    return _BOARD


def net(n):
    if n:
        board().net(n)


def active():
    """True when a Studio is reading the progress file (the log can then be quieter)."""
    return bool(os.environ.get(ENV))


def read(path):
    """The progress a job wrote, or None (missing or unreadable)."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and isinstance(d.get("tasks"), list) else None
