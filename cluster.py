"""Train one from-scratch model on several PCs over the internet (CPU and GPU machines mixed).

    # on the PC that holds the data (data/bin/train.bin, val.bin) and keeps the model:
    python cluster.py coordinator --out runs/my-writer --preset cpu --vocab 8192 --secret <words>
    # on every other PC (Windows: run.bat --join ..., which also installs everything first):
    python cluster.py worker --join http://<coordinator-address>:7861 --secret <words>

How it works (DiLoCo-style local training with periodic averaging, not per-step DDP):
  * Training goes in rounds (default 5 minutes). At the start of a round every worker holds the
    same global weights. Each one trains on its own share of the data for the round's time on
    its own CPU or GPU, with its own AdamW, as many steps as it manages: a GPU simply does more
    steps than a laptop CPU.
  * Each worker then uploads how far its weights moved (one bf16 copy of the model). The
    coordinator averages the moves, weighted by the tokens each worker trained on, and takes one
    Nesterov-momentum "outer" step with the average. That is the next round's global model.
  * Workers download only the change since the round they have (bf16), or the full weights
    after they were away. So the internet carries about one model copy each way per worker per
    round, instead of gradients on every step: a 14M model is ~28 MB, the 110M "small" ~220 MB.
  * Workers may join or leave at any time. A worker that joins mid-round trains for the time that
    is left; one that disappears for 90 s is not waited for; a late upload is dropped. If the
    coordinator restarts (with --resume) the workers reconnect and carry on.
  * Workers that have no data download a share of train.bin from the coordinator once (resumable,
    in 8 MB pieces) and keep it in data/cluster/.

The coordinator writes runs/<name>/ckpt.pt (same format as train.py, so Chat, Export, learn.py and
train.py --resume all work), metrics.jsonl (one "step" per round) and cluster.json (live state
for the Studio). Every request needs the shared secret (X-Secret header). Plain HTTP: over the
open internet use a private network such as Tailscale/ZeroTier/WireGuard, or forward one port.
Weights travel as tensors only (torch.load(weights_only=True)), never as pickled code.
"""
import argparse, hmac, http.client, io, json, math, os, secrets, socket, subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import graceful  # noqa: E402
from core import safeload  # noqa: E402
import netreach  # noqa: E402

DEFAULT_PORT = 7861
GONE_AFTER = 90        # seconds without a heartbeat: the worker left
CHUNK_TOKENS = 4 << 20  # data download piece (8 MB)
PROTOCOL = 1


def pack(obj):
    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def unpack(data):
    return safeload.load_bytes(data)


def inner_lr_at(r, a):
    """Inner learning rate for round r: cosine from --lr to --min-lr over all rounds."""
    p = min(1.0, r / max(1, a.rounds - 1))
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (a.lr - a.min_lr)


# ================================================================ coordinator
class Coordinator:
    def __init__(self, a):
        import torch
        from model.gpt import GPT, GPTConfig, PRESETS

        self.a = a
        self.lock = threading.RLock()
        os.makedirs(a.out, exist_ok=True)
        self.ckpt_path = os.path.join(a.out, "ckpt.pt")
        self.state_path = os.path.join(a.out, "cluster_state.pt")
        self.status_path = os.path.join(a.out, "cluster.json")
        self.train_bin = os.path.join(a.data, "train.bin")
        self.val_bin = os.path.join(a.data, "val.bin")
        if not os.path.exists(self.train_bin):
            sys.exit(f"[cluster] {self.train_bin} is missing: prepare the training data first.")
        self.train_tokens = os.path.getsize(self.train_bin) // 2
        torch.manual_seed(1337)

        cfg = GPTConfig(vocab_size=a.vocab, **PRESETS[a.preset])
        ck = None
        if a.resume and os.path.exists(self.ckpt_path):
            ck = safeload.load(self.ckpt_path)
            cfg = GPTConfig(**ck["config"])
        self.cfg = cfg
        self.model = GPT(cfg)
        self.round, self.best, self.tokens_total = 0, float("inf"), 0
        if ck is not None:
            self.model.load_state_dict(ck["model"])
            self.round, self.best = int(ck.get("step", 0)), ck.get("best_val", float("inf"))
        self.params = dict(self.model.named_parameters())  # tied weights appear once
        for p in self.params.values():
            p.requires_grad_(False)
        self.momentum = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.cluster_id = secrets.token_hex(4)
        if ck is not None and os.path.exists(self.state_path):
            st = safeload.load(self.state_path)
            self.momentum.update(st.get("momentum", {}))
            self.tokens_total = int(st.get("tokens", 0))
            self.cluster_id = st.get("cluster_id", self.cluster_id)
        if ck is not None:
            print(f"[cluster] resumed at round {self.round} (best val {self.best:.3f})", flush=True)
        self.model_bytes = sum(p.numel() * 4 for p in self.params.values())
        self.version = self.round
        self.full_cache = None      # (version, bytes) of the full fp32 weights
        self.update_cache = None    # (from_version, bytes) of the last applied bf16 update
        self.workers = {}           # id -> info
        self.shards = {}            # worker name -> [start, count]  (same share when it rejoins)
        self.cur = None             # the round being trained
        self.phase, self.done = "waiting", False
        self.started = time.time()
        self.eval_thread = None
        self.history = []           # last rounds, for the Studio
        self.log_f = open(os.path.join(a.out, "metrics.jsonl"), "a")
        self.client_worker = None
        self.upnp_state = None if not self.upnp_wanted() else {"pending": True}
        self.keeper = None
        self._addrs = (0, None)
        self.save_lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    def log(self, **kw):
        with self.lock:
            self.log_f.write(json.dumps({"time": time.time(), "max_steps": self.a.rounds, **kw}) + "\n")
            self.log_f.flush()

    def active(self):
        now = time.time()
        return {w: i for w, i in self.workers.items() if now - i["last_seen"] < GONE_AFTER and not i.get("left")}

    def full_weights(self):
        with self.lock:
            if not self.full_cache or self.full_cache[0] != self.version:
                self.full_cache = (self.version, pack({"kind": "full", "version": self.version,
                                                        "tensors": {k: p.detach().clone() for k, p in self.params.items()}}))
            return self.full_cache[1]

    def upnp_wanted(self):
        return self.a.upnp and self.a.host not in ("127.0.0.1", "localhost")

    def reach_addrs(self):
        """Labelled join addresses (internet / tailscale / local), refreshed every minute."""
        t, v = self._addrs
        if v is None or time.time() - t > 60:
            v = netreach.addresses(self.a.port, self.upnp_state)
            self._addrs = (time.time(), v)
        return v

    def upnp_changed(self, st):
        self.upnp_state = st
        self._addrs = (0, None)
        if st.get("ok") and st.get("external_ip") and not st.get("error"):
            print(f"[cluster] router port {self.a.port} opened (UPnP). From other networks join with:  "
                  f"run.bat --join http://{st['external_ip']}:{self.a.port} --secret {self.a.secret}", flush=True)
        else:
            print(f"[cluster] could not open the router port automatically: {st.get('error')}. "
                  "PCs on other networks: use Tailscale, or forward the port on the router.", flush=True)

    def write_status(self):
        addrs = self.reach_addrs()  # may run `tailscale ip`: outside the lock
        with self.lock:
            now = time.time()
            ws = []
            for wid, i in sorted(self.workers.items(), key=lambda x: x[1]["joined"]):
                gone = now - i["last_seen"] >= GONE_AFTER or i.get("left")
                st = "left" if gone else i.get("state", "joined")
                if self.cur and wid in self.cur["deltas"]:
                    st = "sent"
                ws.append({"id": wid, "name": i["name"], "device": i["device"], "device_name": i.get("device_name"),
                           "address": i.get("address"), "state": st, "joined": i["joined"], "last_seen": i["last_seen"],
                           "rounds": i["rounds"], "tokens": i["tokens"], "tokens_per_sec": i.get("tps"),
                           "loss": i.get("loss"), "step": i.get("local_step"), "note": i.get("note")})
            cur = self.cur
            s = {"role": "coordinator", "protocol": PROTOCOL, "pid": os.getpid(), "port": self.a.port,
                 "addresses": [x["ip"] for x in addrs], "reach": {"upnp": self.upnp_state,
                 "addresses": addrs}, "phase": "finished" if self.done else self.phase,
                 "round": self.round, "rounds": self.a.rounds, "round_seconds": self.a.round_seconds,
                 "round_started": cur["started"] if cur else None, "train_until": cur["train_until"] if cur else None,
                 "model_mb": round(self.model_bytes / 1e6, 1), "transfer_mb": round(self.model_bytes / 2e6, 1), "params_m": round(self.model_bytes / 4e6, 1),
                 "tokens_total": self.tokens_total, "train_tokens": self.train_tokens, "best_val": None if self.best == float("inf") else self.best,
                 "min_workers": self.a.min_workers, "workers": ws, "history": self.history[-30:],
                 "updated": now, "started": self.started}
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f)
        os.replace(tmp, self.status_path)

    def save(self):
        import torch
        from dataclasses import asdict

        with self.save_lock:
            self._save(torch, asdict)

    def _save(self, torch, asdict):
        with self.lock:
            model_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            mom = {k: v.clone() for k, v in self.momentum.items()}
            meta = {"step": self.round, "best_val": self.best}
        args = {k: v for k, v in vars(self.a).items() if k != "secret" and isinstance(v, (int, float, str, bool, type(None)))}
        tmp = self.ckpt_path + ".tmp"
        torch.save({"model": model_sd, "opt": None, "config": asdict(self.cfg), **meta, "args": args,
                    "cluster": True}, tmp)
        os.replace(tmp, self.ckpt_path)
        tmp = self.state_path + ".tmp"
        torch.save({"momentum": mom, "tokens": self.tokens_total, "cluster_id": self.cluster_id}, tmp)
        os.replace(tmp, self.state_path)

    # ---------------------------------------------------------------- rounds
    def start_round(self):
        now = time.time()
        with self.lock:
            act = self.active()
            self.cur = {"n": self.round, "started": now, "train_until": now + self.a.round_seconds,
                        "deadline": now + self.a.round_seconds + max(120, self.a.round_seconds),
                        "lr": inner_lr_at(self.round, self.a), "participants": set(act), "deltas": {}}
            self.phase = "training"
        print(f"[cluster] round {self.round + 1}/{self.a.rounds} started with {len(act)} worker(s)", flush=True)

    def round_finished(self):
        with self.lock:
            c = self.cur
            act = self.active()
            waiting = [w for w in c["participants"] if w in act and w not in c["deltas"]]
            if c["deltas"] and not waiting and time.time() >= c["started"] + 5:
                return True
            if time.time() > c["deadline"]:
                return True
            if not c["deltas"] and not any(w in act for w in c["participants"]) and time.time() > c["train_until"]:
                return True
        return False

    def close_round(self):
        import torch

        with self.lock:
            c, self.cur = self.cur, None
            deltas = c["deltas"]
            if not deltas:
                print(f"[cluster] round {c['n'] + 1}: no worker sent its result; repeating it", flush=True)
                self.phase = "waiting"
                return False
            self.phase = "averaging"
            total = sum(d["tokens"] for d in deltas.values())
            mu, lr = self.a.outer_momentum, self.a.outer_lr
            upd = {}
            for k, p in self.params.items():
                g = sum(d["delta"][k].float() * (d["tokens"] / total) for d in deltas.values())
                m = self.momentum[k]
                m.mul_(mu).add_(g)
                step = g + mu * m if self.a.nesterov else m
                u = (-lr * step).to(torch.bfloat16)
                # Apply the rounded update, so a worker that adds the same bf16 update to its copy
                # holds bit-for-bit the coordinator's weights.
                p.add_(u.float())
                upd[k] = u
            self.update_cache = (self.version, pack({"kind": "update", "from": self.version, "version": self.version + 1,
                                                     "tensors": upd}))
            self.version += 1
            self.round += 1
            self.tokens_total += total
            secs = time.time() - c["started"]
            loss = sum(d["loss"] * d["tokens"] for d in deltas.values()) / total
            for wid, d in deltas.items():
                if wid in self.workers:
                    w = self.workers[wid]
                    w["rounds"] += 1
                    w["tokens"] += d["tokens"]
            self.history.append({"round": self.round, "workers": len(deltas), "tokens": total, "seconds": secs,
                                 "loss": loss, "time": time.time()})
            self.phase = "waiting"
        tps = total / max(1e-6, secs)
        print(f"[cluster] round {self.round}/{self.a.rounds} | {len(deltas)} worker(s) | {total:,} tokens | "
              f"loss {loss:.3f} | {tps:,.0f} tok/s", flush=True)
        self.log(kind="train", step=self.round, loss=loss, lr=c["lr"], tokens_per_sec=tps, workers=len(deltas))
        self.save()
        if self.round % self.a.eval_every == 0 or self.round >= self.a.rounds:
            self.evaluate_async(final=self.round >= self.a.rounds)
        return True

    def evaluate_async(self, final=False):
        import torch
        from model.gpt import GPT

        if not os.path.exists(self.val_bin):
            return
        if self.eval_thread and self.eval_thread.is_alive():
            if not final:
                return
            self.eval_thread.join()
        with self.lock:
            snap = GPT(self.cfg)
            snap.load_state_dict(self.model.state_dict())
            r = self.round

        def run():
            import numpy as np

            data = np.memmap(self.val_bin, dtype=np.uint16, mode="r")
            rng = np.random.default_rng(7)
            blk, losses = self.cfg.block_size, []
            snap.eval()
            with torch.no_grad():
                for _ in range(self.a.eval_iters):
                    ix = rng.integers(0, len(data) - blk - 1, self.a.eval_batch)
                    x = torch.from_numpy(np.stack([data[i:i + blk].astype(np.int64) for i in ix]))
                    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + blk].astype(np.int64) for i in ix]))
                    losses.append(snap(x, y)[1].item())
            vl = sum(losses) / len(losses)
            print(f"== round {r} val loss {vl:.3f} (ppl {math.exp(min(vl, 20)):.1f})", flush=True)
            self.log(kind="val", step=r, loss=vl)
            with self.lock:
                improved = vl < self.best
                self.best = min(self.best, vl)
            if improved:
                self.save()

        self.eval_thread = threading.Thread(target=run, daemon=True)
        self.eval_thread.start()
        if final:
            self.eval_thread.join()

    # ---------------------------------------------------------------- requests
    def join(self, body, address):
        name = (str(body.get("name") or "worker")[:40]).strip() or "worker"
        with self.lock:
            wid = secrets.token_hex(4)
            if body.get("protocol") != PROTOCOL:
                raise ValueError("this PC's copy of Text LLM is a different version than the coordinator's; "
                                 "update both copies from GitHub (git pull) and start again")
            if body.get("vocab_ok") is False:
                raise ValueError("worker data does not fit this model's vocabulary")
            want = int(max(1, float(body.get("data_mb") or 0)) * 1e6 // 2) if body.get("data_mb") else 0
            shard = None
            if want:
                if name in self.shards and self.shards[name][1] == min(want, self.train_tokens):
                    shard = self.shards[name]
                else:
                    count = min(want, self.train_tokens)
                    start = (len(self.shards) * count) % max(1, self.train_tokens - count + 1)
                    shard = self.shards[name] = [int(start), int(count)]
            self.workers[wid] = {"name": name, "device": body.get("device", "?"), "device_name": body.get("device_name"),
                                 "address": address, "joined": time.time(), "last_seen": time.time(), "rounds": 0,
                                 "tokens": 0, "state": "joined"}
            others = sum(1 for i in self.active().values() if i["name"] == name) - 1
            if others > 0:
                self.workers[wid]["note"] = "same name as another worker"
        print(f"[cluster] {name} joined from {address} ({body.get('device_name') or body.get('device')})", flush=True)
        a = self.a
        return {"worker": wid, "protocol": PROTOCOL, "cluster_id": self.cluster_id,
                "config": {k: getattr(self.cfg, k) for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout")},
                "train": {"weight_decay": a.weight_decay, "warmup": a.warmup, "clip": 1.0},
                "data": {"tokens": self.train_tokens, "shard": shard}, "round_seconds": a.round_seconds}

    def task(self, wid):
        with self.lock:
            w = self.workers.get(wid)
            if not w:
                return None
            w["last_seen"] = time.time()
            if self.done:
                w["state"] = "done"
                return {"action": "done", "version": self.version}
            c = self.cur
            if c and wid not in c["deltas"]:
                left = c["train_until"] - time.time()
                if wid not in c["participants"] and left >= max(20, 0.2 * self.a.round_seconds):
                    c["participants"].add(wid)  # joined mid-round: trains for the time that is left
                if wid in c["participants"] and left > 0:
                    return {"action": "train", "round": c["n"], "version": self.version, "seconds": left, "lr": c["lr"]}
            act = len(self.active())
            note = (f"waiting for workers ({act}/{self.a.min_workers})" if act < self.a.min_workers and not c
                    else "waiting for the next round")
            return {"action": "wait", "version": self.version, "retry": 2 if c is None else 3, "note": note}

    def weights(self, have):
        with self.lock:
            if have == self.version:
                return pack({"kind": "same", "version": self.version})
            if self.update_cache and have == self.update_cache[0]:
                return self.update_cache[1]
        return self.full_weights()

    def delta(self, wid, rnd, payload):
        with self.lock:
            w = self.workers.get(wid)
            if not w:
                return 404, {"error": "unknown worker; join again"}
            w["last_seen"] = time.time()
            c = self.cur
            if not c or c["n"] != rnd:
                return 409, {"error": "stale: that round is already closed"}
        d = unpack(payload)
        delta, tokens = d.get("delta"), int(d.get("tokens", 0))
        if not isinstance(delta, dict) or set(delta) != set(self.params) or tokens <= 0:
            return 400, {"error": "result does not match the model"}
        for k, p in self.params.items():
            if tuple(delta[k].shape) != tuple(p.shape) or not bool(delta[k].float().isfinite().all()):
                return 400, {"error": f"bad tensor {k}"}
        with self.lock:
            c = self.cur
            if not c or c["n"] != rnd:
                return 409, {"error": "stale: that round is already closed"}
            c["deltas"][wid] = {"delta": delta, "tokens": tokens, "loss": float(d.get("loss", 0.0))}
            w.update(tps=d.get("tokens_per_sec"), loss=float(d.get("loss", 0.0)), state="sent")
        return 200, {"ok": True}

    def heartbeat(self, wid, body):
        with self.lock:
            w = self.workers.get(wid)
            if not w:
                return False
            w["last_seen"] = time.time()
            w.pop("left", None)
            for k in ("state", "tps", "loss", "local_step", "note"):
                if k in body:
                    w[k] = body[k]
        return True

    def leave(self, wid):
        with self.lock:
            if wid in self.workers:
                self.workers[wid]["left"] = True
                print(f"[cluster] {self.workers[wid]['name']} left", flush=True)

    def data(self, start, count):
        count = max(0, min(count, CHUNK_TOKENS, self.train_tokens - start))
        with open(self.train_bin, "rb") as f:
            f.seek(start * 2)
            return f.read(count * 2)

    # ---------------------------------------------------------------- main loop
    def serve(self):
        coord = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authed(self):
                if hmac.compare_digest(self.headers.get("X-Secret", ""), coord.a.secret):
                    return True
                time.sleep(0.5)  # slow down guessing
                self.reply(403, {"error": "wrong secret"})
                return False

            def body(self, limit):
                n = int(self.headers.get("Content-Length") or 0)
                if n > limit:
                    raise ValueError("request too large")
                data, got = bytearray(), 0
                while got < n:
                    piece = self.rfile.read(min(1 << 20, n - got))
                    if not piece:
                        break
                    data += piece
                    got += len(piece)
                return bytes(data)

            def route(self, method):
                u = urllib.parse.urlparse(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
                if u.path == "/ping":
                    return self.reply(200, {"textllm_cluster": PROTOCOL})
                if not self.authed():
                    return
                try:
                    wid = q.get("worker", "")
                    if method == "POST" and u.path == "/join":
                        return self.reply(200, coord.join(json.loads(self.body(1 << 16) or b"{}"), self.client_address[0]))
                    if method == "GET" and u.path == "/task":
                        t = coord.task(wid)
                        return self.reply(200, t) if t else self.reply(404, {"error": "unknown worker; join again"})
                    if method == "GET" and u.path == "/weights":
                        if wid not in coord.workers:
                            return self.reply(404, {"error": "unknown worker; join again"})
                        return self.reply(200, coord.weights(int(q.get("have", -1))), "application/octet-stream")
                    if method == "POST" and u.path == "/delta":
                        code, r = coord.delta(wid, int(q.get("round", -1)), self.body(coord.model_bytes + (8 << 20)))
                        return self.reply(code, r)
                    if method == "POST" and u.path == "/heartbeat":
                        ok = coord.heartbeat(wid, json.loads(self.body(1 << 16) or b"{}"))
                        return self.reply(200, {"ok": True}) if ok else self.reply(404, {"error": "unknown worker; join again"})
                    if method == "POST" and u.path == "/leave":
                        coord.leave(wid)
                        return self.reply(200, {"ok": True})
                    if method == "GET" and u.path == "/data":
                        if wid not in coord.workers:
                            return self.reply(404, {"error": "unknown worker; join again"})
                        return self.reply(200, coord.data(int(q.get("start", 0)), int(q.get("count", CHUNK_TOKENS))),
                                          "application/octet-stream")
                    if method == "GET" and u.path == "/status":
                        return self.reply(200, json.load(open(coord.status_path)))
                    self.reply(404, {"error": "no such endpoint"})
                except (ValueError, KeyError, RuntimeError) as e:
                    self.reply(400, {"error": str(e)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                self.route("GET")

            def do_POST(self):
                self.route("POST")

        ThreadingHTTPServer.allow_reuse_address = True
        try:
            self.httpd = ThreadingHTTPServer((self.a.host, self.a.port), H)
        except OSError as e:
            sys.exit(f"[cluster] can't listen on port {self.a.port}: {e}")
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def run(self):
        a = self.a
        self.serve()
        print(f"[cluster] coordinator {self.cfg.n_layer}x{self.cfg.n_embd} model "
              f"({self.model_bytes / 4e6:.1f}M params, {self.model_bytes / 2e6:.0f} MB per transfer), "
              f"{self.train_tokens:,} training tokens", flush=True)
        for x in netreach.addresses(a.port):
            print(f"[cluster] workers join ({x['note']}) with:  run.bat --join {x['url']} --secret {a.secret}", flush=True)
        if self.upnp_wanted():
            self.keeper = netreach.PortKeeper(a.port, on_change=self.upnp_changed)
        if a.local_worker:
            cmd = [sys.executable, os.path.abspath(__file__), "worker", "--join", f"http://127.0.0.1:{a.port}",
                   "--name", f"{socket.gethostname()[:30]} (this PC)", "--data", a.data,
                   "--device", a.local_device, "--batch", str(a.batch), "--grad-accum", str(a.grad_accum)]
            # same process group, so Stop in the Studio reaches it too
            self.client_worker = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "TEXTLLM_CLUSTER_SECRET": a.secret})
        last_status = 0
        try:
            while self.round < a.rounds:
                if time.time() - last_status > 2:
                    self.write_status()
                    last_status = time.time()
                if self.cur is None:
                    if len(self.active()) >= a.min_workers:
                        self.start_round()
                elif self.round_finished():
                    self.close_round()
                    self.write_status()
                time.sleep(0.5)
            self.done = True
            if self.eval_thread:
                self.eval_thread.join()
            self.write_status()
            print(f"[cluster] finished {self.round} rounds, {self.tokens_total:,} tokens", flush=True)
            end = time.time() + 20  # let the workers hear that training is over
            while time.time() < end and any(i.get("state") not in ("done",) for i in self.active().values()):
                time.sleep(1)
        except KeyboardInterrupt:
            print("[cluster] stopping; the last finished round is saved", flush=True)
            self.phase = "stopped"
            self.done = True
        finally:
            self.write_status()
            if self.client_worker:
                try:
                    self.client_worker.wait(30)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    self.client_worker.kill()
            if self.keeper:
                self.keeper.stop()  # close the router port again
            self.httpd.shutdown()


# ================================================================ worker
class Gone(Exception):
    """The coordinator forgot this worker (it restarted): join again."""


class Client:
    def __init__(self, url, secret, timeout=600):
        self.url, self.secret, self.timeout = url.rstrip("/"), secret, timeout

    def call(self, method, path, body=None, raw=False, retries=None, ctype="application/json", say=print):
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            if not isinstance(body, bytes):
                ctype = "application/json"
        delay, n = 2, 0
        while True:
            req = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"X-Secret": self.secret, "Content-Type": ctype})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = r.read()
                    return out if raw else json.loads(out or b"{}")
            except urllib.error.HTTPError as e:
                msg = e.read()[:300].decode("utf-8", "replace")
                if e.code == 404 and "join again" in msg:
                    raise Gone()
                if e.code in (400, 403, 409):
                    raise RuntimeError(f"{e.code}: {msg}")
                err = f"HTTP {e.code}"
            except (urllib.error.URLError, OSError, ConnectionError, http.client.HTTPException) as e:  # incl. IncompleteRead
                err = str(getattr(e, "reason", e))
            n += 1
            if retries is not None and n > retries:
                raise ConnectionError(err)
            say(f"[worker] coordinator not reachable ({err}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(60, delay * 2)


class Worker:
    def __init__(self, a):
        self.a = a
        self.client = Client(a.join, a.secret)
        self.status = {"role": "worker", "coordinator": a.join, "state": "connecting", "pid": os.getpid(),
                       "name": a.name, "rounds": 0, "tokens": 0, "started": time.time()}
        self.wid = None
        self.stop = threading.Event()

    def say(self, msg):
        print(msg, flush=True)
        self.status["note"] = msg.replace("[worker] ", "")
        self.write_status()

    def write_status(self):
        if not self.a.status_file:
            return
        self.status["updated"] = time.time()
        tmp = self.a.status_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.status, f)
            os.replace(tmp, self.a.status_file)
        except OSError:
            pass

    def heartbeat_loop(self):
        while not self.stop.wait(15):
            if not self.wid:
                continue
            try:
                self.client.call("POST", f"/heartbeat?worker={self.wid}",
                                 {"state": self.status["state"], "tps": self.status.get("tokens_per_sec"),
                                  "loss": self.status.get("loss"), "local_step": self.status.get("local_step")},
                                 retries=0, say=lambda *_: None)
            except Exception:
                pass

    def do_join(self, devname):
        import torch

        info = self.client.call("POST", "/join", {"name": self.a.name, "device": self.dev, "device_name": devname,
                                                  "data_mb": None if self.local_data else self.a.data_mb,
                                                  "torch": torch.__version__, "protocol": PROTOCOL}, say=self.say)
        self.wid = info["worker"]
        return info

    def fetch_data(self, info):
        """Download this worker's share of train.bin once, resuming after a dropped connection."""
        start, count = info["data"]["shard"]
        cache = os.environ.get("TEXTLLM_CLUSTER_CACHE") or os.path.join(ROOT, "data", "cluster")
        d = os.path.join(cache, f"{info['cluster_id']}-{start}-{count}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "train.bin")
        if os.path.exists(path) and os.path.getsize(path) == count * 2:
            return path
        part = path + ".part"
        have = os.path.getsize(part) // 2 if os.path.exists(part) else 0
        with open(part, "ab") as f:
            f.truncate(have * 2)
            while have < count:
                n = min(CHUNK_TOKENS, count - have)
                try:
                    chunk = self.client.call("GET", f"/data?worker={self.wid}&start={start + have}&count={n}", raw=True, say=self.say)
                except Gone:
                    self.do_join(self.status.get("device_name"))
                    continue
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk) // 2
                self.status.update(state="downloading data", data_progress=have / count)
                self.say(f"[worker] downloaded {have * 2 / 1e6:.0f} of {count * 2 / 1e6:.0f} MB of training text")
        os.replace(part, path)
        return path

    def run(self):
        import numpy as np
        import torch

        import devices
        from model.gpt import GPT, GPTConfig
        from train import Loader
        from writing import hf

        a = self.a
        self.dev = devices.pick(a.device)
        kind = devices.kind(self.dev)
        devname = devices.describe(self.dev)
        dtype = devices.train_dtype(self.dev)
        autocast = devices.autocast(self.dev, dtype)
        if kind == "cpu":
            hf.set_threads()
            autocast = devices.cpu_autocast(self.dev) or autocast
        scaler = devices.grad_scaler(self.dev, dtype)
        self.status.update(device=self.dev, device_name=devname)
        self.local_data = a.data and os.path.exists(os.path.join(a.data, "train.bin"))
        self.say(f"[worker] {a.name} on {devname}: joining {a.join}")
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        info = self.do_join(devname)
        cfg = GPTConfig(**info["config"])
        tr = info["train"]
        if self.local_data:
            path = os.path.join(a.data, "train.bin")
            sample = np.memmap(path, dtype=np.uint16, mode="r")[: 1 << 20]
            if len(sample) and int(sample.max()) >= cfg.vocab_size:
                raise SystemExit(f"[worker] {path} was made with a bigger tokenizer than the model's "
                                 f"{cfg.vocab_size}-token vocabulary. Use the coordinator's data (leave --data empty).")
        else:
            path = self.fetch_data(info)
        torch.manual_seed(int.from_bytes(os.urandom(4), "little"))
        loader = Loader(path, a.batch, cfg.block_size, self.dev, rank=int(info["worker"], 16) % 100000, world=1)
        model = GPT(cfg).to(self.dev)
        from train import make_optimizer

        params = dict(model.named_parameters())
        opt = make_optimizer(model, 1e-4, tr["weight_decay"], self.dev)
        glob, version, local_step = None, -1, 0  # glob: the global weights, fp32 on the CPU
        self.status.update(state="ready", tokens_per_round=a.batch * a.grad_accum * cfg.block_size)
        self.say(f"[worker] joined as {self.wid}; {model.num_params() / 1e6:.1f}M params, "
                 f"{a.batch * a.grad_accum * cfg.block_size:,} tokens per step")
        try:
            while True:
                try:
                    t = self.client.call("GET", f"/task?worker={self.wid}", say=self.say)
                    if t["action"] == "done":
                        self.status["state"] = "done"
                        self.say("[worker] the coordinator finished training. Thank you!")
                        break
                    if t["action"] == "wait":
                        self.status["state"] = "waiting"
                        if self.status.get("note") != t.get("note"):
                            self.say(f"[worker] {t.get('note', 'waiting')}")
                        else:
                            self.write_status()
                        time.sleep(t.get("retry", 3))
                        continue
                    if version != t["version"]:
                        self.status["state"] = "downloading weights"
                        self.write_status()
                        t0 = time.time()
                        w = unpack(self.client.call("GET", f"/weights?worker={self.wid}&have={version}", raw=True, say=self.say))
                        if w["kind"] == "full":
                            glob = {k: v.float() for k, v in w["tensors"].items()}
                        elif w["kind"] == "update":
                            for k, u in w["tensors"].items():
                                glob[k].add_(u.float())
                        version = w["version"]
                        if w["kind"] != "same":
                            print(f"[worker] got the round {version} model ({w['kind']}) in {time.time() - t0:.1f}s", flush=True)
                    with torch.no_grad():
                        for k, p in params.items():
                            p.copy_(glob[k])
                    if version != t["version"]:
                        continue  # a round closed while downloading
                    # ---- train for the time this round has left
                    self.status.update(state="training", note=f"training round {t['round'] + 1} for {t['seconds']:.0f}s")
                    self.write_status()
                    end = time.time() + max(5.0, t["seconds"] - a.reserve)
                    steps, tokens, loss_sum, t0 = 0, 0, 0.0, time.time()
                    model.train()
                    while steps == 0 or time.time() < end:
                        lr = t["lr"] * min(1.0, (local_step + 1) / max(1, tr["warmup"]))
                        for g in opt.param_groups:
                            g["lr"] = lr
                        lsum = 0.0
                        for _ in range(a.grad_accum):
                            x, y = loader.next()
                            with autocast:
                                loss = model(x, y)[1] / a.grad_accum
                            scaler.scale(loss).backward()
                            lsum += loss.item()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip"])
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                        steps += 1
                        local_step += 1
                        tokens += a.batch * a.grad_accum * cfg.block_size
                        loss_sum += lsum
                        if steps % 10 == 0:
                            self.status.update(local_step=local_step, loss=loss_sum / steps,
                                               tokens_per_sec=tokens / (time.time() - t0))
                            self.write_status()
                    secs = time.time() - t0
                    with torch.no_grad():
                        delta = {k: (glob[k] - p.detach().float().cpu()).to(torch.bfloat16) for k, p in params.items()}
                    payload = pack({"delta": delta, "tokens": tokens, "steps": steps, "loss": loss_sum / steps,
                                    "tokens_per_sec": tokens / secs})
                    self.status.update(state="uploading", local_step=local_step, loss=loss_sum / steps,
                                       tokens_per_sec=tokens / secs)
                    self.write_status()
                    print(f"[worker] round {t['round'] + 1}: {steps} steps, {tokens:,} tokens, loss {loss_sum / steps:.3f}, "
                          f"{tokens / secs:,.0f} tok/s; sending {len(payload) / 1e6:.1f} MB", flush=True)
                    try:
                        self.client.call("POST", f"/delta?worker={self.wid}&round={t['round']}", payload,
                                         ctype="application/octet-stream", say=self.say)
                        self.status["rounds"] += 1
                        self.status["tokens"] += tokens
                        self.status["state"] = "sent"
                        self.write_status()
                    except RuntimeError as e:
                        if "409" in str(e):
                            self.say("[worker] that round closed before the upload finished; joining the next one")
                        else:
                            raise
                except Gone:
                    self.say("[worker] the coordinator restarted; joining again")
                    self.do_join(devname)
                    version = -1
        except KeyboardInterrupt:
            self.status["state"] = "stopped"
            self.say("[worker] stopping")
        finally:
            self.stop.set()
            if self.wid and self.status["state"] != "done":
                try:
                    self.client.call("POST", f"/leave?worker={self.wid}", {}, retries=0, say=lambda *_: None)
                except Exception:
                    pass
            if self.status["state"] not in ("done", "stopped"):
                self.status["state"] = "stopped"
            self.write_status()


# ================================================================ command line
def main():
    graceful.install()
    from model.gpt import PRESETS

    ap = argparse.ArgumentParser(description="Train one model on several PCs over the network.")
    sub = ap.add_subparsers(dest="role", required=True)
    c = sub.add_parser("coordinator", help="hold the model and the data; workers connect here")
    c.add_argument("--out", default="runs/cluster")
    c.add_argument("--data", default="data/bin")
    c.add_argument("--preset", default="cpu", choices=PRESETS)
    c.add_argument("--vocab", type=int, default=8192)
    c.add_argument("--rounds", type=int, default=200)
    c.add_argument("--round-minutes", type=float, default=5.0, help="local training time per round")
    c.add_argument("--lr", type=float, default=1e-3, help="workers' learning rate (cosine over the rounds)")
    c.add_argument("--min-lr", type=float, default=1e-4)
    c.add_argument("--warmup", type=int, default=200, help="each worker's first steps ramp the learning rate up")
    c.add_argument("--weight-decay", type=float, default=0.1)
    c.add_argument("--outer-lr", type=float, default=0.7, help="step size of the averaged update (DiLoCo: 0.7)")
    c.add_argument("--outer-momentum", type=float, default=0.9)
    c.add_argument("--no-nesterov", dest="nesterov", action="store_false")
    c.add_argument("--min-workers", type=int, default=1, help="workers needed before a round starts")
    c.add_argument("--eval-every", type=int, default=1, help="rounds between validation checks")
    c.add_argument("--eval-iters", type=int, default=20)
    c.add_argument("--eval-batch", type=int, default=8)
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=DEFAULT_PORT)
    c.add_argument("--secret", default=os.environ.get("TEXTLLM_CLUSTER_SECRET", ""),
                   help="shared password the workers must send (made up if empty; or TEXTLLM_CLUSTER_SECRET)")
    c.add_argument("--no-upnp", dest="upnp", action="store_false",
                   help="don't ask the router to open the port for PCs on other networks")
    c.add_argument("--local-worker", action="store_true", help="also train on this PC")
    c.add_argument("--local-device", default="auto")
    c.add_argument("--batch", type=int, default=16, help="micro-batch of the local worker")
    c.add_argument("--grad-accum", type=int, default=4, help="grad accumulation of the local worker")
    c.add_argument("--resume", action="store_true")

    w = sub.add_parser("worker", help="lend this PC's CPU or GPU to a coordinator")
    w.add_argument("--join", required=True, help="coordinator address, e.g. http://203.0.113.5:7861")
    w.add_argument("--secret", default=os.environ.get("TEXTLLM_CLUSTER_SECRET", ""),
                   help="the coordinator's secret (or TEXTLLM_CLUSTER_SECRET)")
    w.add_argument("--name", default=socket.gethostname()[:40])
    w.add_argument("--device", default="auto", help="auto (GPU if there is one), cuda, cuda:1, mps, xpu or cpu")
    w.add_argument("--batch", type=int, default=0, help="micro-batch (0 = 16 on a GPU, 8 on a CPU)")
    w.add_argument("--grad-accum", type=int, default=4)
    w.add_argument("--data", default="", help="use this PC's data folder (train.bin, same tokenizer) instead of downloading")
    w.add_argument("--data-mb", type=float, default=200, help="size of the share of training text to download")
    w.add_argument("--reserve", type=float, default=0, help="stop training this many seconds early (slow upload)")
    w.add_argument("--status-file", default="", help="write live state here (used by the Studio)")
    a = ap.parse_args()

    if a.role == "coordinator":
        if not a.secret:
            a.secret = "-".join(secrets.token_hex(4) for _ in range(4))  # 128 bits: safe on the open internet
        a.round_seconds = max(10.0, a.round_minutes * 60)
        a.out, a.data = os.path.abspath(a.out), os.path.abspath(a.data)
        Coordinator(a).run()
    else:
        if not a.secret:
            ap.error("worker needs --secret (shown on the coordinator's Training → Cluster page)")
        if not a.join.startswith(("http://", "https://")):
            a.join = "http://" + a.join
        if urllib.parse.urlparse(a.join).port is None and a.join.count(":") < 2:
            a.join = a.join.rstrip("/") + f":{DEFAULT_PORT}"
        if not a.batch:
            import devices

            a.batch = 8 if devices.kind(devices.pick(a.device)) == "cpu" else 16
        try:
            Worker(a).run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
