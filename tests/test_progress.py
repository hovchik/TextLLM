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


def test_a_smaller_source_keeps_its_real_total(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "data"))
    import build_corpus as bc

    p = bc.Progress(str(tmp_path / "src"), "Tiny catalog", 1000, unit="books", pos=0)
    p.set_total(40)                  # the catalog has only 40 books
    with p:
        p.write("x" * 100)
        assert p.task.total == 40 and p.bar.total == 40


def test_concurrent_writers_leave_valid_json(tmp_path):
    path = tmp_path / "p.json"
    b = progress.Board(str(path))
    t = b.task("a", "A", 10**9, "bytes").start()

    def work():
        for _ in range(300):
            b.net(10)
            t.add(1)
            b.touch(force=True)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    b.touch(force=True)
    d = progress.read(str(path))
    assert d is not None and d["net_bytes"] == 8 * 300 * 10


# ---------------------------------------------------------------- tokenizer and encoding (local steps)
def _old_encode(raw, tok_path, val_frac, fim_rate=0.0, seed=0):
    """The encoder before progress was added (one tokenizer call per shard), as the reference output."""
    from tokenizers import Tokenizer

    from coding import fim
    from coding.langs import is_code_doc
    from tokenizer.train_tokenizer import EOT, shards

    tok = Tokenizer.from_file(tok_path)
    eot, ids = tok.token_to_id(EOT), fim.token_ids(tok)
    out = {"train": [], "val": []}
    rng, frng = __import__("random").Random(0), __import__("random").Random(seed + 1)
    for path in shards(raw):
        docs = [d.strip() for d in open(path, encoding="utf-8").read().split(EOT) if d.strip()]
        plan, pieces = [], []
        for d in docs:
            if fim_rate > 0 and is_code_doc(d) and frng.random() < fim_rate:
                pre, mid, suf = fim.split(d, frng)
                plan.append((len(pieces), 3))
                pieces += [pre, suf, mid]
            else:
                plan.append((len(pieces), 1))
                pieces.append(d)
        enc = [e.ids for e in tok.encode_batch(pieces)]
        for start, n in plan:
            if n == 1:
                seq = enc[start]
            else:
                pre, suf, mid = enc[start:start + 3]
                seq = [ids["prefix"], *pre, ids["suffix"], *suf, ids["middle"], *mid]
            out["val" if rng.random() < val_frac else "train"] += seq + [eot]
    return out


def _corpus(d, files=3, docs=60, code=False):
    import random

    r = random.Random(1)
    for i in range(files):
        sub = d / ("src" if i % 2 else "books")
        sub.mkdir(parents=True, exist_ok=True)
        parts = []
        for j, t in enumerate(sample_text(docs, seed=i)):
            if code and j % 2:
                t = "def f%d(x):\n    return x + %d\n\nclass A:\n    pass\n" % (j, r.randint(0, 9)) * 3
            parts.append(t)
        (sub / f"shard_{i:05d}.txt").write_text("\n<|endoftext|>\n".join(parts) + "\n<|endoftext|>\n")


@pytest.mark.parametrize("fim_rate", [0.0, 0.5])
def test_encode_reports_progress_and_matches_the_old_output(tmp_path, monkeypatch, fim_rate):
    import numpy as np

    from tokenizer import train_tokenizer as tt

    raw = tmp_path / "raw"
    _corpus(raw, code=fim_rate > 0)
    tok = str(tmp_path / "bpe.json")
    tt.train(str(raw), 600, tok, 0, code=fim_rate > 0)
    monkeypatch.setattr(tt, "BATCH_CHARS", 3000)        # many batches per shard
    path = tmp_path / "p.json"
    monkeypatch.setattr(progress, "_BOARD", progress.Board(str(path)))
    tt.encode(str(raw), tok, str(tmp_path / "bin"), 0.1, fim_rate)
    want = _old_encode(str(raw), tok, 0.1, fim_rate)
    for split in ("train", "val"):
        got = np.fromfile(tmp_path / "bin" / f"{split}.bin", dtype=np.uint16).tolist()
        assert got == want[split], split
    d = json.loads(path.read_text())
    assert d["kind"] == "work" and d["title"] == "Encoding the corpus"
    (t,) = d["tasks"]
    size = sum(f.stat().st_size for f in raw.rglob("*.txt"))
    assert (t["state"], t["pct"], t["done"], t["total"], t["verb"]) == ("done", 100.0, size, size, "Encoding")
    assert "3 files" in t["note"] and "M tokens" in t["note"]


def test_tokenizer_build_reports_progress_and_is_unchanged(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from tokenizer import train_tokenizer as tt

    raw = tmp_path / "raw"
    _corpus(raw)
    path = tmp_path / "p.json"
    monkeypatch.setattr(progress, "_BOARD", progress.Board(str(path)))
    tt.train(str(raw), 500, str(tmp_path / "new.json"), 0, style="gpt2", lang_balance=1.0, check_mb=0)
    # the same settings trained the old way (the library reading the files itself)
    old = Tokenizer(models.BPE())
    old.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    old.train(tt.shards(str(raw)), trainers.BpeTrainer(vocab_size=500, special_tokens=[tt.EOT], show_progress=False,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    assert Tokenizer.from_file(str(tmp_path / "new.json")).get_vocab() == old.get_vocab()
    d = json.loads(path.read_text())
    assert d["kind"] == "work" and [t["state"] for t in d["tasks"]] == ["done", "done"]
    assert d["tasks"][0]["done"] == d["tasks"][0]["total"] > 0 and d["overall"]["pct"] == 100
    # the default (modern) build also reports its quality check
    monkeypatch.setattr(progress, "_BOARD", progress.Board(str(path)))
    tt.train(str(raw), 500, str(tmp_path / "v2.json"), 0, check_mb=0.5)
    d = json.loads(path.read_text())
    assert [t["state"] for t in d["tasks"]] == ["done", "done", "done"] and d["overall"]["pct"] == 100


def test_a_failed_encode_says_so(tmp_path, monkeypatch):
    from tokenizer import train_tokenizer as tt

    raw = tmp_path / "raw"
    _corpus(raw, files=1)
    tok = str(tmp_path / "bpe.json")
    tt.train(str(raw), 500, tok, 0)
    (raw / "books" / "shard_00000.txt").write_bytes(b"\xff\xfe not utf-8")
    path = tmp_path / "p.json"
    monkeypatch.setattr(progress, "_BOARD", progress.Board(str(path)))
    with pytest.raises(UnicodeDecodeError):
        tt.encode(str(raw), tok, str(tmp_path / "bin"), 0.1)
    assert json.loads(path.read_text())["tasks"][0]["state"] == "failed"
