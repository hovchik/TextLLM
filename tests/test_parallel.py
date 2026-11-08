"""Several training processes on one PC (core/parallel.py, train.py --procs/--threads)."""
import importlib, json, os

import numpy as np
import pytest

from conftest import ROOT, run_script
from core import parallel


def test_plan_splits_threads_and_batch(monkeypatch):
    monkeypatch.delenv("TEXTLLM_THREADS", raising=False)
    monkeypatch.setattr(parallel, "physical_cores", lambda: 8)
    monkeypatch.setattr(parallel, "load_bench", lambda path=None: {})
    assert parallel.plan(0, 0, "cpu") == (1, 8)          # auto without a speed test: one process
    assert parallel.plan(2, 0, "cpu") == (2, 4)
    assert parallel.plan(4, 6, "cpu") == (4, 1)          # --threads caps the total
    assert parallel.plan(16, 0, "cpu") == (8, 1)         # never more processes than threads
    assert parallel.plan(4, 0, "mps") == (1, 8)          # one Apple GPU: one process
    assert parallel.plan(0, 0, "cuda", gpus=2)[0] == 2   # one per NVIDIA GPU
    assert parallel.plan(8, 0, "cuda", gpus=2)[0] == 2
    monkeypatch.setenv("TEXTLLM_THREADS", "3")
    assert parallel.plan(1, 0, "cpu") == (1, 3)
    assert [parallel.split_batch(16, n) for n in (1, 2, 3, 4)] == [16, 8, 6, 4]


def test_auto_uses_the_speed_test(tmp_path, monkeypatch):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"cpu": {"best": 2, "results": {"1": 900, "2": 700}}}))
    monkeypatch.setattr(parallel, "BENCH_PATH", str(p))
    assert parallel.auto_procs("cpu", preset="cpu") == 2
    assert parallel.auto_procs("cpu", preset="tiny") == 1


def test_ms_per_step_skips_the_first_logged_step():
    log = "device cpu\nstep 10 | loss 9 | lr 1e-3 | 2000 ms/step\nstep 20 | loss 9 | lr 1e-3 | 500 ms/step\n" \
          "step 30 | loss 9 | lr 1e-3 | 700 ms/step\n"
    assert parallel._ms_per_step(log) == 600
    assert parallel._ms_per_step("nothing") is None


def _random_bin(d, vocab=512, n=200_000):
    d.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, vocab, n, dtype=np.uint16).tofile(d / "train.bin")
    rng.integers(0, vocab, n // 10, dtype=np.uint16).tofile(d / "val.bin")


def test_two_processes_train_and_resume(tmp_path):
    pytest.importorskip("torch.distributed")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("this PyTorch has no torch.distributed")
    _random_bin(tmp_path / "bin")
    out = tmp_path / "run"
    base = ["train.py", "--data", str(tmp_path / "bin"), "--out", str(out), "--preset", "tiny", "--vocab", "512",
            "--batch", "4", "--grad-accum", "2", "--eval-every", "10", "--eval-iters", "2", "--warmup", "2",
            "--procs", "2", "--threads", "2"]
    log = run_script(*base, "--max-steps", "10")
    assert "2 training processes on this PC, 1 CPU threads each, micro-batch 2 each (total 4)" in log
    assert "× 2 processes" in log and "== step 10 val loss" in log
    # tokens/step counts both processes: 2 rows × 2 accum × 2 processes × 256
    assert "2,048 tokens/step" in log
    from core import safeload

    ck = safeload.load(str(out / "ckpt.pt"))
    assert ck["step"] == 10 and not any(k.startswith("module.") for k in ck["model"])
    # the checkpoint continues in a single process (another PC, or --procs 1)
    log = run_script(*base[:-4], "--procs", "1", "--max-steps", "20", "--resume")
    assert "resumed at step 10" in log and "== step 20 val loss" in log


def test_studio_passes_and_replaces_speed_flags(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app
    from studio import auth, db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "studio.db"))
    db.init()
    importlib.reload(auth)
    runs = tmp_path / "runs"
    monkeypatch.setattr(app, "RUNS", str(runs))
    seen = []
    monkeypatch.setattr(app, "_launch", lambda kind, name, cmd, *a, **kw: seen.append((kind, cmd)) or {"ok": True})
    c = TestClient(app.app)
    assert c.post("/api/auth/setup", json={"username": "admin", "password": "secret123"}).status_code == 200

    # a run from another PC, saved with 4 processes and 3 threads: Resume uses this PC's choice
    d = runs / "moved"
    d.mkdir(parents=True)
    (d / "tokenizer.json").write_text("{}")
    (d / "meta.json").write_text(json.dumps({"type": "scratch"}))
    (d / "train_cmd.json").write_text(json.dumps(["train.py", "--out", "runs/moved", "--data", "data/bin",
                                                  "--max-steps", "100", "--procs", "4", "--threads", "3"]))
    r = c.post("/api/train", json={"mode": "scratch", "name": "moved", "resume": True, "procs": 2, "steps": 100})
    assert r.status_code == 200, r.text
    cmd = seen[-1][1]
    assert cmd.count("--procs") == 1 and cmd[cmd.index("--procs") + 1] == "2"
    assert "--threads" not in cmd and "--resume" in cmd

    # LoRA gets --threads (no --procs: one process, PyTorch threads)
    (d / "train_cmd.json").write_text(json.dumps(["finetune/finetune_lora.py", "--out", "runs/moved", "--max-steps", "5"]))
    r = c.post("/api/train", json={"mode": "lora", "name": "moved", "resume": True, "threads": 3, "procs": 2})
    assert r.status_code == 200, r.text
    cmd = seen[-1][1]
    assert "--procs" not in cmd and cmd[cmd.index("--threads") + 1] == "3"

    r = c.post("/api/train/speedtest", json={"preset": "tiny", "batch": 8})
    assert r.status_code == 200 and seen[-1][0] == "speedtest"
    assert seen[-1][1][1:4] == ["-m", "core.parallel", "--preset"]
    assert c.post("/api/train/speedtest", json={"preset": "nope"}).status_code == 400
    p = c.get("/api/train/parallel").json()
    assert p["cores"] >= 1 and isinstance(p["bench"], dict)


def test_speed_test_end_to_end(tmp_path):
    pytest.importorskip("torch.distributed")
    path = str(tmp_path / "bench.json")
    out = parallel.bench("tiny", vocab=512, batch=4, steps=20, candidates=[1, 2], threads=2, path=path,
                         log=lambda m: None)
    assert set(out["results"]) == {1, 2} and out["best"] in (1, 2)
    assert json.load(open(path))["tiny"]["best"] == out["best"]
