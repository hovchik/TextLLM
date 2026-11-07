"""Studio internals: the database, job stopping, the status poll and the feedback gate."""
import json, os, subprocess, sys, time

import pytest

from conftest import ROOT


@pytest.fixture
def db(tmp_path, monkeypatch):
    from studio import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "studio.db"))
    db.init()
    return db


def test_db_round_trip_and_shared_connection(db):
    cid = db.new_conversation("hello", "m")
    uid = db.add_message(cid, "user", "hi", fmt="free")
    db.add_message(cid, "assistant", "hello there", model="m", prompt_id=uid)
    assert [m["content"] for m in db.history(cid)] == ["hi", "hello there"]
    assert db.conn() is db.conn()
    db.set_settings(auto_threshold=3)
    assert db.get_settings()["auto_threshold"] == 3


def test_db_rolls_back_a_failed_statement(db):
    with pytest.raises(Exception):
        db.ex("INSERT INTO messages(conversation_id, role, content, created) VALUES(1, 'user', NULL, 0)")
    assert db.q("SELECT COUNT(*) n FROM messages", one=True)["n"] == 0
    db.ex("INSERT INTO settings(key, value) VALUES('x', '1')")  # the connection still works
    assert db.get_settings()["x"] == 1


def test_db_init_migrates_an_old_database(tmp_path, monkeypatch):
    import sqlite3

    from studio import db

    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.executescript("CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT, model TEXT, created REAL, updated REAL);"
                    "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role TEXT, content TEXT,"
                    " model TEXT, prompt_id INTEGER, created REAL, rating INTEGER, comment TEXT, correction TEXT,"
                    " rated_at REAL, trained_in TEXT);")
    c.close()
    monkeypatch.setattr(db, "DB_PATH", p)
    db.init()
    cols = [r["name"] for r in db.conn().execute("PRAGMA table_info(messages)")]
    assert {"format", "fields", "check_score"} <= set(cols)


def test_stop_returns_at_once_and_kills_a_job_that_ignores_it(tmp_path, monkeypatch):
    from studio import jobs

    monkeypatch.setattr(jobs, "LOCK_PATH", str(tmp_path / "lock"))
    monkeypatch.setattr(jobs, "HISTORY_PATH", str(tmp_path / "jobs.jsonl"))
    stubborn = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('up', flush=True); time.sleep(60)"
    jobs.launch("test", "stubborn", [sys.executable, "-c", stubborn], str(tmp_path / "log.txt"))
    for _ in range(50):
        if "up" in open(tmp_path / "log.txt").read():
            break
        time.sleep(0.1)
    t0 = time.time()
    jobs.stop(grace=1)
    assert time.time() - t0 < 5, "stop() blocked the request"
    jobs.JOB["proc"].wait(15)  # killed once the grace period is over
    assert not jobs.running()


def test_status_poll_caches_the_disk_scan(tmp_path, monkeypatch):
    from studio import jobs

    raw = tmp_path / "data" / "raw" / "custom"
    raw.mkdir(parents=True)
    (raw / "a.txt").write_text("x" * 1000)
    (tmp_path / "tokenizer").mkdir()
    tokp = tmp_path / "tokenizer" / "bpe.json"
    tokp.write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    monkeypatch.setattr(jobs, "ROOT", str(tmp_path))
    monkeypatch.setitem(jobs._raw_scan, "v", None)
    s = jobs.data_status()
    assert (s["raw_files"], s["custom_files"], s["vocab"]) == (1, 1, 2)
    (raw / "b.txt").write_text("y")
    assert jobs.data_status()["raw_files"] == 1          # cached for a few seconds
    jobs._raw_scan["t"] = 0
    assert jobs.data_status()["raw_files"] == 2          # rescanned after the TTL
    tokp.write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1, "c": 2}}}))
    os.utime(tokp, (time.time() + 5, time.time() + 5))
    assert jobs.data_status()["vocab"] == 3              # a new tokenizer is noticed at once


def test_run_summary_ignores_rejected_and_capped_evaluations(tmp_path):
    from studio import jobs

    run = tmp_path / "r"
    (run / "eval").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"type": "lora", "base": "b"}))
    rows = [{"time": 1, "score": 70, "quick": True}, {"time": 2, "score": 40, "quick": True, "rejected": True},
            {"time": 3, "score": 55, "quick": True, "max_tokens": 300}, {"time": 4, "score": 99, "partial": True}]
    (run / "eval" / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert jobs.run_summary(str(run))["eval_score"] == 70


# ---------------------------------------------------------------- feedback gate
def gate_run(tmp_path):
    run = tmp_path / "fb-run"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "w.bin").write_text("old weights")
    (run / "meta.json").write_text(json.dumps({"type": "lora", "base": "b"}))
    data = tmp_path / "fb.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}], "chosen": "hello"}) + "\n")
    return run, data


def run_gate(monkeypatch, run, data, after):
    import feedback_train as ft

    def fake_train(a, meta, examples):
        (run / "adapter" / "w.bin").write_text("new weights")

    def fake_quick(r, fresh=False):
        if not fresh:
            return {"score": 80, "per_format": {"x_post": 80}}
        h = run / "eval" / "history.jsonl"
        h.parent.mkdir(exist_ok=True)
        with open(h, "a") as f:
            f.write(json.dumps({"time": time.time(), "score": 50, "quick": True, "max_tokens": 300}) + "\n")
        return after()

    monkeypatch.setattr(ft, "train_lora", fake_train)
    monkeypatch.setattr(ft, "quick_score", fake_quick)
    monkeypatch.setattr(sys, "argv", ["feedback_train.py", "--run", str(run), "--data", str(data), "--gate"])
    ft.main()


def test_gate_rejection_restores_adapter_and_marks_its_evaluation(tmp_path, monkeypatch):
    run, data = gate_run(tmp_path)
    with pytest.raises(SystemExit) as e:
        run_gate(monkeypatch, run, data, lambda: {"score": 50, "per_format": {"x_post": 50}})
    assert e.value.code == 3
    assert (run / "adapter" / "w.bin").read_text() == "old weights"
    rows = [json.loads(l) for l in open(run / "eval" / "history.jsonl")]
    assert all(r.get("rejected") for r in rows), "the undone adapter's score would become the baseline"


def test_gate_restores_adapter_when_stopped_during_the_check(tmp_path, monkeypatch):
    run, data = gate_run(tmp_path)

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_gate(monkeypatch, run, data, interrupted)
    assert (run / "adapter" / "w.bin").read_text() == "old weights"


def test_app_imports_without_loading_torch():
    code = "import sys, app; print('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_api_requires_sign_in():
    from fastapi.testclient import TestClient

    import app

    c = TestClient(app.app)
    assert c.get("/api/auth/state").status_code == 200
    assert c.get("/api/status").status_code == 401
