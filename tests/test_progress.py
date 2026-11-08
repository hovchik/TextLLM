"""Download progress: the board in core/progress.py, build_corpus.py writing it for every source,
and the Studio's job status reading it back.

    python -m pytest tests/test_progress.py
"""
import io, json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import ROOT, run_script, sample_text
from core import progress


def test_board_writes_tasks_overall_and_eta(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    b = progress.Board(str(path))
    b.plan([("a", "Books", 10, "books"), ("b", "Web", 1000, "bytes")], title="Downloading")
    t = b.task("a").start(done=2)
    clock = [1000.0]
    monkeypatch.setattr(progress.time, "time", lambda: clock[0])
    b.started = 1000.0
    t.rate = progress._Rate()
    t.rate.add(1000.0, 2)
    b._f0 = None
    b.snapshot()                    # overall 10% at the start of the run
    clock[0] = 1010.0
    t.add(3)                        # 5 of 10 books after 10 s: 0.3 books/s, 5 left
    b.net(4096)
    b.touch(force=True)
    d = json.loads(path.read_text())
    a, w = d["tasks"]
    assert (a["label"], a["state"], a["pct"]) == ("Books", "running", 50.0)
    assert a["eta"] == pytest.approx(5 / 0.3, rel=0.01)
    assert (w["state"], w["pct"]) == ("waiting", 0)
    assert d["overall"]["pct"] == 25.0 and d["overall"]["current"] == "Books"
    assert d["overall"]["eta"] is not None and d["net_bytes"] == 4096
    t.finish()
    b.task("b").start().finish("skipped", "nothing to do")
    d = b.snapshot()
    assert d["overall"]["pct"] == 100 and d["overall"]["done"] == 2
    assert "2/2 sources done" in b.summary() and "4.0 KB downloaded" in b.summary()


def test_board_without_a_file_writes_nothing(tmp_path):
    b = progress.Board(None)
    b.task("x", "X", 5, "books").start().add(5)
    assert b.snapshot()["tasks"][0]["pct"] == 100
    assert not list(tmp_path.iterdir())


def test_formatting():
    assert progress.fmt_bytes(1536) == "1.5 KB"
    assert progress.fmt_eta(59) == "59s" and progress.fmt_eta(3700) == "1h 01m" and progress.fmt_eta(None) == "—"


# ---------------------------------------------------------------- build_corpus.py end to end
def _parquet(n=300):
    import pyarrow as pa, pyarrow.parquet as pq

    texts = [" ".join(sample_text(3, seed=i)) for i in range(n)]
    buf = io.BytesIO()
    pq.write_table(pa.table({"text": texts}), buf, row_group_size=50)
    return buf.getvalue()


@pytest.fixture
def fake_hf():
    """Enough of the Hugging Face API for hf_parquet: dataset info, file tree, ranged downloads."""
    files = {"sample/10BT/000.parquet": _parquet(), "sample/10BT/001.parquet": _parquet()}
    sha = "0123456789abcdef0123456789abcdef01234567"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/api/datasets/HuggingFaceFW/fineweb-edu":
                return self._send(200, json.dumps({"sha": sha}).encode())
            if p.startswith("/api/datasets/HuggingFaceFW/fineweb-edu/tree/"):
                rows = [{"type": "file", "path": k, "size": len(v)} for k, v in files.items()]
                return self._send(200, json.dumps(rows).encode())
            m = re.match(rf"/datasets/HuggingFaceFW/fineweb-edu/resolve/{sha}/(.+)", p)
            if m and m.group(1) in files:
                data = files[m.group(1)]
                rng = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if rng:
                    a = int(rng.group(1))
                    b = int(rng.group(2)) if rng.group(2) else len(data) - 1
                    return self._send(206, data[a:b + 1], "application/octet-stream",
                                      [("Content-Range", f"bytes {a}-{b}/{len(data)}")])
                return self._send(200, data, "application/octet-stream")
            self._send(404, b"{}")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.mark.parametrize("mode", ["stream", "download"])
def test_build_corpus_reports_progress(tmp_path, fake_hf, mode):
    prog = tmp_path / "progress.json"
    env = {**os.environ, "HF_ENDPOINT": fake_hf, progress.ENV: str(prog)}
    r = subprocess.run([sys.executable, "data/build_corpus.py", "--out", str(tmp_path / "raw"), "--gutenberg-max", "0",
                        "--fineweb-gb", "0.0001", "--fineweb-mode", mode, "--cache", str(tmp_path / "cache"),
                        "--retries", "2"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "[overall 0%" in r.stdout and "Done. [overall 100%" in r.stdout
    d = json.loads(prog.read_text())
    assert d["title"] == "Downloading the training corpus"
    (t,) = d["tasks"]
    assert (t["label"], t["state"], t["unit"], t["pct"]) == ("FineWeb-Edu", "done", "bytes", 100)
    assert t["total"] == int(0.0001 * 1024**3) and t["done"] >= t["total"]
    assert d["net_bytes"] > 1000 and d["overall"]["pct"] == 100

    # a second run has nothing to fetch: the line says so instead of staying at "waiting"
    r = subprocess.run(r.args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    (t,) = json.loads(prog.read_text())["tasks"]
    assert (t["state"], t["note"]) == ("done", "already downloaded")


def test_build_corpus_plan_lists_every_source():
    sys.path.insert(0, os.path.join(ROOT, "data"))
    import argparse

    import build_corpus as bc

    a = argparse.Namespace(gutenberg_max=10, pg19_books=0, standard_ebooks=5, common_corpus_gb=0, archive_books=0,
                           cosmopedia_gb=1.0, cosmopedia_subsets="stories,wikihow", wikipedia_gb=0.5,
                           wikipedia_lang=" HY ", common_pile_gb=0, common_pile_subsets="news", fineweb_gb=0.25)
    labels = [(k, total, unit) for k, _, total, unit in bc.plan(a)]
    assert labels == [("Gutenberg", 10, "books"), ("Standard Ebooks", 5, "books"),
                      ("Cosmopedia stories", 512 * 1024**2, "bytes"), ("Cosmopedia wikihow", 512 * 1024**2, "bytes"),
                      ("Wikipedia hy", 512 * 1024**2, "bytes"), ("FineWeb-Edu", 256 * 1024**2, "bytes")]


# ---------------------------------------------------------------- Studio job status
def test_job_status_includes_progress_and_marks_a_stopped_source(tmp_path, monkeypatch):
    from studio import jobs

    monkeypatch.setattr(jobs, "PROGRESS_PATH", str(tmp_path / "_progress.json"))
    monkeypatch.setattr(jobs, "LOCK_PATH", str(tmp_path / "_job.lock"))
    monkeypatch.setattr(jobs, "HISTORY_PATH", str(tmp_path / "jobs.jsonl"))
    (tmp_path / "_progress.json").write_text("stale from the last job")
    script = ("import os, sys; sys.path.insert(0, os.getcwd()); from core import progress\n"
              "b = progress.board(); b.plan([('a', 'Books', 10, 'books'), ('b', 'Web', 5, 'bytes')], title='T')\n"
              "t = b.task('a').start(); t.add(4); b.touch(force=True)\n"
              "import time; time.sleep(30)\n")
    jobs.launch("data", "download", [sys.executable, "-c", script], str(tmp_path / "log.txt"))
    try:
        for _ in range(100):
            p = jobs.status()["progress"]
            if p and p["tasks"][0]["done"] == 4:
                break
            time.sleep(0.1)
        assert p["title"] == "T" and p["tasks"][0]["state"] == "running" and p["overall"]["count"] == 2
    finally:
        jobs.stop(grace=5, wait=True)
    p = jobs.status()["progress"]
    assert p["tasks"][0]["state"] == "stopped" and p["overall"]["eta"] is None
    assert p["tasks"][1]["state"] == "waiting"
