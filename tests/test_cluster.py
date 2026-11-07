"""cluster.py on one machine: a coordinator and two workers (one downloads its data share, one
uses local data), a worker joining late, and the checkpoint working with train.py --resume."""
import json, os, socket, subprocess, sys, time

import pytest
import torch

from conftest import ROOT, run_script
from test_scratch_pipeline import write_bins

SECRET = "test-secret"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start(*args, env=None):
    return subprocess.Popen([sys.executable, "cluster.py", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env={**os.environ, "TEXTLLM_THREADS": "1", **(env or {})})


@pytest.fixture(scope="module")
def cluster(bpe, tmp_path_factory):
    d = tmp_path_factory.mktemp("cluster")
    data = str(d / "bin")
    vocab = write_bins(bpe, data)
    out = str(d / "run")
    port = free_port()
    coord = start("coordinator", "--out", out, "--data", data, "--preset", "tiny", "--vocab", str(vocab),
                  "--rounds", "3", "--round-minutes", str(4 / 60), "--min-workers", "2", "--warmup", "1",
                  "--eval-iters", "2", "--eval-batch", "2", "--host", "127.0.0.1", "--port", str(port),
                  "--secret", SECRET)
    url = f"http://127.0.0.1:{port}"
    common = ["--secret", SECRET, "--device", "cpu", "--batch", "2", "--grad-accum", "1"]
    status = str(d / "w1.json")
    w1 = start("worker", "--join", url, "--name", "pc-download", "--data-mb", "0.01", "--status-file", status, *common,
               env={"TEXTLLM_CLUSTER_CACHE": str(d / "cache")})
    w2 = start("worker", "--join", url, "--name", "pc-local", "--data", data, *common)
    outs = {}
    for name, p in (("coord", coord), ("w1", w1), ("w2", w2)):
        try:
            outs[name] = p.communicate(timeout=240)[0]
        except subprocess.TimeoutExpired:
            for q in (coord, w1, w2):
                q.kill()
            raise AssertionError("cluster did not finish: " + "\n".join(f"{k}: {v[-2000:]}" for k, v in outs.items()))
    return {"out": out, "data": data, "vocab": vocab, "outs": outs, "codes": (coord.returncode, w1.returncode, w2.returncode),
            "status": status}


def test_cluster_trains_all_rounds(cluster):
    o = cluster["outs"]
    assert cluster["codes"] == (0, 0, 0), o
    assert "round 3/3" in o["coord"] and "2 worker(s)" in o["coord"], o["coord"]
    assert "finished training" in o["w1"] and "finished training" in o["w2"]
    assert "downloaded" in o["w1"]        # took a share of train.bin from the coordinator
    assert "downloaded" not in o["w2"]    # used its own data
    assert "(update)" in o["w1"]          # later rounds only fetch the bf16 change


def test_checkpoint_and_metrics(cluster):
    ck = torch.load(os.path.join(cluster["out"], "ckpt.pt"), map_location="cpu")  # weights_only default, like Chat
    assert ck["step"] == 3 and ck["config"]["vocab_size"] == cluster["vocab"] and "secret" not in ck["args"]
    rows = [json.loads(l) for l in open(os.path.join(cluster["out"], "metrics.jsonl"))]
    assert [r["step"] for r in rows if r["kind"] == "train"] == [1, 2, 3]
    assert any(r["kind"] == "val" for r in rows) and all(r["max_steps"] == 3 for r in rows)
    st = json.load(open(os.path.join(cluster["out"], "cluster.json")))
    assert st["phase"] == "finished" and st["round"] == 3 and len(st["workers"]) == 2
    assert all(w["rounds"] >= 2 for w in st["workers"])
    ws = json.load(open(cluster["status"]))
    assert ws["state"] == "done" and ws["rounds"] >= 2


def test_train_py_continues_a_cluster_checkpoint(cluster, tmp_path):
    import shutil

    out = str(tmp_path / "cont")
    shutil.copytree(cluster["out"], out)
    log = run_script("train.py", "--data", cluster["data"], "--out", out, "--preset", "tiny", "--vocab", str(cluster["vocab"]),
                     "--batch", "2", "--grad-accum", "1", "--max-steps", "5", "--warmup", "1", "--eval-every", "5",
                     "--eval-iters", "2", "--device", "cpu", "--resume")
    assert "resumed at step 3" in log


def test_wrong_secret_is_refused(cluster, tmp_path):
    from cluster import Client

    port = free_port()
    coord = start("coordinator", "--out", str(tmp_path / "r"), "--data", cluster["data"], "--preset", "tiny",
                  "--vocab", str(cluster["vocab"]), "--rounds", "1", "--host", "127.0.0.1", "--port", str(port),
                  "--secret", SECRET)
    try:
        good = Client(f"http://127.0.0.1:{port}", SECRET)
        for _ in range(100):
            try:
                good.call("GET", "/ping", retries=0, say=lambda *_: None)
                break
            except ConnectionError:
                time.sleep(0.2)
        with pytest.raises(RuntimeError, match="403"):
            Client(f"http://127.0.0.1:{port}", "nope").call("POST", "/join", {"name": "x"}, retries=0)
        info = good.call("POST", "/join", {"name": "late", "device": "cpu"})
        assert info["config"]["vocab_size"] == cluster["vocab"] and info["data"]["shard"] is None
    finally:
        coord.terminate()
        coord.wait(30)


def test_client_retries_a_cut_off_download():
    """A response that ends before its Content-Length (connection dropped mid-body) is retried."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cluster import Client

    hits = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(1)
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * (10 if len(hits) == 1 else 100))
            if len(hits) == 1:
                self.close_connection = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import cluster

        sleep, cluster.time.sleep = cluster.time.sleep, lambda s: None
        try:
            out = Client(f"http://127.0.0.1:{srv.server_address[1]}", "s").call("GET", "/data", raw=True, retries=3,
                                                                                 say=lambda *_: None)
        finally:
            cluster.time.sleep = sleep
        assert out == b"x" * 100 and len(hits) == 2
    finally:
        srv.shutdown()
