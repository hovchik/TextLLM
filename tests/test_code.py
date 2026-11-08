"""The copilot (code) model: code sources, secret scrubbing, .gitignore, licence checks, FIM data and completion."""
import io, json, os, random, tarfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import EOT, ROOT, run_script

PY_FILE = '''import os


def add(a, b):
    """Add two numbers."""
    return a + b


def greet(name):
    return f"hello {name}"


class Counter:
    def __init__(self):
        self.n = 0

    def inc(self):
        self.n += 1
        return self.n
'''
JS_FILE = """export function chunk(list, size) {
  const out = [];
  for (let i = 0; i < list.length; i += size) {
    out.push(list.slice(i, i + size));
  }
  return out;
}
"""


# ---------------------------------------------------------------- pieces
def test_language_detection_and_header():
    from coding import langs as L

    assert L.detect("src/app.py")[0] == "Python"
    assert L.detect("Dockerfile")[0] == "Dockerfile"
    assert L.detect("tool", "#!/usr/bin/env python3\nprint(1)")[0] == "Python"
    assert L.detect("notes.bin") is None
    doc = L.header("proj/x.css", L.detect("x.css")[1]) + "\nbody {}\n"
    assert doc.startswith("/* path: proj/x.css */") and L.is_code_doc(doc) and L.doc_path(doc) == "proj/x.css"
    assert L.doc_lang("# path: a/b.py\nx = 1") == "Python"
    assert not L.is_code_doc("Once upon a time")
    assert L.wanted("C++", L.parse_langs("python, c++")) and not L.wanted("Markdown", set())


def test_gitignore_rules():
    from coding.gitignore import Ignore

    r = Ignore(["*.log", "build/", "/secret.txt", "docs/**/*.tmp", "!keep.log", "**/cache"])
    assert r.ignored("a/b/x.log") and not r.ignored("a/keep.log")
    assert r.ignored("build", is_dir=True) and not r.ignored("build")  # dir-only rule
    assert r.ignored("secret.txt") and not r.ignored("sub/secret.txt")  # anchored
    assert r.ignored("docs/a/b/c.tmp") and not r.ignored("other/c.tmp")
    assert r.ignored("x/y/cache", is_dir=True)
    r.add_lines(["*.gen.py"], base="pkg")
    assert r.ignored("pkg/sub/m.gen.py") and not r.ignored("m.gen.py")


def test_secrets_are_scrubbed():
    from coding.secrets_scan import REDACTED, scrub, skip_file

    assert skip_file(".env") and skip_file(".env.production") and skip_file("id_rsa") and skip_file("prod.pem")
    assert not skip_file(".env.example") and not skip_file("settings.py")
    src = ('GITHUB = "ghp_' + "a" * 36 + '"\n'
           'aws = "AKIAABCDEFGHIJKLMNOP"\n'
           'password = "hunter2hunter2"\n'
           'api_key: "sk-' + "b" * 40 + '"\n'
           'DB = "postgres://admin:s3cretpw@db:5432/app"\n'
           'export API_TOKEN=abcdef123456\n'
           'password = os.environ["PW"]\n'
           'token = "your_token_here"\n')
    out, n = scrub(src)
    assert "ghp_" not in out and "AKIA" not in out and "hunter2" not in out and "s3cretpw" not in out
    assert "abcdef123456" not in out and "sk-bbbb" not in out
    assert 'password = os.environ["PW"]' in out and "your_token_here" in out  # not secrets
    assert n >= 6 and REDACTED in out
    assert scrub("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n")[0] is None


def test_fim_split_and_format():
    from coding import fim

    doc = "# path: p/a.py\n" + PY_FILE
    rng = random.Random(3)
    for _ in range(50):
        pre, mid, suf = fim.split(doc, rng)
        assert pre + mid + suf == doc and pre.startswith("# path: p/a.py\n")
    assert fim.psm("a", "c", "b") == "<|fim_prefix|>a<|fim_suffix|>c<|fim_middle|>b"
    assert fim.split_cursor("x = <cursor>\ny") == ("x = ", "\ny") and fim.split_cursor("x") == ("x", "")
    assert "<|endoftext|>" not in fim.defuse("tok = '<|endoftext|>'")


# ---------------------------------------------------------------- local projects
@pytest.fixture
def project(tmp_path):
    p = tmp_path / "myapp"
    (p / "src").mkdir(parents=True)
    (p / "src" / "util.py").write_text(PY_FILE)
    (p / "src" / "chunk.js").write_text(JS_FILE)
    (p / "src" / "config.py").write_text('API_KEY = "sk-' + "z" * 40 + '"\n' + PY_FILE)
    (p / ".env").write_text("SECRET=supersecretvalue\n")
    (p / "node_modules" / "lib").mkdir(parents=True)
    (p / "node_modules" / "lib" / "index.js").write_text(JS_FILE + "// vendored\n")
    (p / "generated").mkdir()
    (p / "generated" / "out.py").write_text(PY_FILE + "# x\n")
    (p / ".gitignore").write_text("generated/\n*.local.py\n")
    (p / "settings.local.py").write_text(PY_FILE + "# local\n")
    (p / "min.js").write_text("var a=1;" * 300 + "\nvar b;\nvar c;\n")
    (p / "README.md").write_text("# My app\n\nSome docs here for people.\nMore.\n")
    (p / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, to any person")
    return p


def read_docs(d):
    docs = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".txt"):
                docs += [x.strip() for x in open(os.path.join(root, f), encoding="utf-8").read().split(EOT) if x.strip()]
    return docs


def test_local_projects_follow_gitignore_and_drop_secrets(project, tmp_path):
    out = tmp_path / "code"
    run_script("data/build_code.py", "--local", str(project), "--out", str(out), "--langs", "")
    docs = read_docs(out)
    paths = sorted(d.split("\n", 1)[0] for d in docs)
    assert paths == ["# path: myapp/src/config.py", "# path: myapp/src/util.py", "// path: myapp/src/chunk.js"]
    assert not any("supersecretvalue" in d or "sk-zzzz" in d for d in docs)
    info = json.load(open(next((out / "local").iterdir()) / "info.json"))
    assert info["files"] == 3 and info["license"] == "mit" and info["redacted"] >= 1
    assert info["skipped"]["secret_file"] == 1 and info["skipped"]["minified"] == 1
    assert info["skipped"]["gitignore"] >= 1
    # reading the folder again replaces its code instead of adding a second copy
    run_script("data/build_code.py", "--local", str(project), "--out", str(out), "--langs", "python")
    assert len(read_docs(out)) == 2


# ---------------------------------------------------------------- GitHub (a fake API on localhost)
def tarball(name, files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-abc123/{path}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


REPOS = {
    "alice/good": {"license": {"spdx_id": "MIT"}, "files": {"LICENSE": "MIT License\nPermission is hereby granted",
                                                           "lib/add.py": PY_FILE, "web/chunk.js": JS_FILE,
                                                           "node_modules/x/i.js": JS_FILE + "//x\n", ".env": "A=b"}},
    "bob/copyleft": {"license": {"spdx_id": "GPL-3.0"}, "files": {"LICENSE": "GNU GPL", "a.py": PY_FILE}},
    "carol/nolicense": {"license": None, "files": {"a.py": PY_FILE + "# c\n"}},
    "dave/nofile": {"license": {"spdx_id": "Apache-2.0"}, "files": {"a.py": PY_FILE + "# d\n"}},
}


QUERIES = []


@pytest.fixture
def fake_github():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/search/repositories":
                from urllib.parse import parse_qs, urlparse

                qs = parse_qs(urlparse(self.path).query)
                QUERIES.append(qs.get("q", [""])[0])
                page = int(qs.get("page", ["1"])[0])
                items = [{"full_name": k, "license": v["license"], "default_branch": "main", "size": 10,
                          "stargazers_count": 100, "html_url": f"https://github.com/{k}",
                          "pushed_at": v.get("pushed_at", "2026-01-01T00:00:00Z")} for k, v in REPOS.items()]
                return self.send(200, {"items": items if page == 1 else []})
            if path.startswith("/repos/"):
                full = path[len("/repos/"):]
                if full not in REPOS:
                    return self.send(404, {"message": "Not Found"})
                v = REPOS[full]
                return self.send(200, {"full_name": full, "license": v["license"], "default_branch": "main", "size": 10,
                                       "pushed_at": v.get("pushed_at", "2026-01-01T00:00:00Z")})
            if "/tar.gz/" in path:
                full = path.strip("/").split("/tar.gz/")[0]
                return self.send(200, tarball(full.split("/")[1], REPOS[full]["files"]), "application/gzip")
            self.send(404, {})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_github_keeps_only_permissive_licensed_repos(fake_github, tmp_path):
    import subprocess, sys

    out = tmp_path / "code"
    prog = tmp_path / "progress.json"
    env = {**os.environ, "GITHUB_API": fake_github, "GITHUB_CODELOAD": fake_github, "TEXTLLM_PROGRESS": str(prog)}
    r = subprocess.run([sys.executable, "data/build_code.py", "--github-repos", "10", "--langs", "python,javascript",
                        "--out", str(out), "--retries", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    p = json.load(open(prog))  # the Studio's download progress: one line for the search, finished
    (t,) = p["tasks"]
    assert (t["label"], t["done"], t["total"], t["state"]) == ("GitHub: search", 1, 10, "done")
    assert p["net_bytes"] > 0 and p["overall"]["pct"] == 100
    docs = read_docs(out)
    assert sorted(d.split("\n", 1)[0] for d in docs) == ["# path: alice/good/lib/add.py", "// path: alice/good/web/chunk.js"]
    st = json.load(open(out / "github" / "progress.json"))
    assert st["done"] == ["alice/good"]
    assert set(st["rejected"]) == {"bob/copyleft", "carol/nolicense", "dave/nofile"}
    attr = [json.loads(l) for l in open(out / "github" / "attribution.jsonl")]
    assert attr[0]["repo"] == "alice/good" and attr[0]["license"] == "MIT" and attr[0]["files"] == 2
    assert (out / "github" / "licenses" / "alice__good.license").exists()
    assert "alice/good" in (out / "ATTRIBUTION.md").read_text()
    # a named repository with a copyleft licence is refused too; a second run skips what is done
    r = subprocess.run([sys.executable, "data/build_code.py", "--github-repo", "bob/copyleft", "--github-repo", "alice/good",
                        "--out", str(out), "--retries", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "already downloaded" in r.stdout and len(read_docs(out)) == 2


def test_github_bigger_total_continues_on_the_same_search_page(fake_github, tmp_path):
    import subprocess, sys

    out = tmp_path / "code"
    env = {**os.environ, "GITHUB_API": fake_github, "GITHUB_CODELOAD": fake_github}
    for n in ("1", "2"):
        if n == "2":  # stopped partway through page 1: the next run reads page 1 again
            assert json.load(open(out / "github" / "progress.json"))["pages"].get("python|mit|50", 1) == 1
        r = subprocess.run([sys.executable, "data/build_code.py", "--github-repos", n, "--langs", "python", "--licenses", "mit,apache-2.0",
                            "--out", str(out), "--retries", "1"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
    # the fake search returns every repo on page 1 only; the second run must still find alice/good
    assert json.load(open(out / "github" / "progress.json"))["done"] == ["alice/good"]


def test_hf_source_creates_its_folder(tmp_path, monkeypatch):
    from data import build_code, build_corpus

    seen = {}
    monkeypatch.setattr(build_corpus, "hf_parquet", lambda outdir, *a, **kw: seen.setdefault("out", outdir))
    build_code.hf_code(str(tmp_path / "code"), 0.1, set(), ["mit"], 256, "stream", 1, "cache", False)
    assert os.path.isdir(seen["out"]) and os.path.exists(os.path.join(seen["out"], "hashes.list"))


def test_only_permissive_licences_can_be_chosen(tmp_path):
    import subprocess, sys

    r = subprocess.run([sys.executable, "data/build_code.py", "--github-repos", "1", "--licenses", "mit,gpl-3.0",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "not a permissive licence: gpl-3.0" in r.stderr


# ---------------------------------------------------------------- from scratch: tokenizer + FIM encoding + chat
@pytest.fixture(scope="module")
def code_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("codecorpus")
    raw = d / "code" / "local" / "p"
    raw.mkdir(parents=True)
    docs = []
    for i in range(60):
        docs.append(f"# path: p/m{i}.py\n" + PY_FILE.replace("add", f"add{i}"))
        docs.append(f"// path: p/c{i}.js\n" + JS_FILE.replace("chunk", f"chunk{i}"))
    (raw / "shard_00000.txt").write_text(f"\n{EOT}\n".join(docs) + f"\n{EOT}\n", encoding="utf-8")
    return d


def test_code_tokenizer_and_fim_encoding(code_corpus):
    import numpy as np
    from tokenizers import Tokenizer

    tok = str(code_corpus / "bpe-code.json")
    run_script("tokenizer/train_tokenizer.py", "--code", "--raw", str(code_corpus / "code"), "--vocab", "600",
               "--out", tok, "--sample-files", "0")
    t = Tokenizer.from_file(tok)
    from coding import fim

    ids = fim.token_ids(t)
    assert ids and t.token_to_id(EOT) == 0
    out = run_script("tokenizer/train_tokenizer.py", "--encode", "--raw", str(code_corpus / "code"), "--tok", tok,
                     "--bin", str(code_corpus / "bin"), "--fim-rate", "0.5", "--val-frac", "0.05")
    assert "rearranged for fill-in-the-middle" in out
    arr = np.fromfile(code_corpus / "bin" / "train.bin", dtype=np.uint16)
    n_pre = int((arr == ids["prefix"]).sum())
    assert 20 < n_pre < 110 and n_pre == int((arr == ids["middle"]).sum())


def test_scratch_copilot_completes_between_prefix_and_suffix(code_corpus, tmp_path):
    import torch
    from model.gpt import GPT, GPTConfig
    from tokenizers import Tokenizer

    tok_p = code_corpus / "bpe-code.json"
    if not tok_p.exists():
        pytest.skip("needs test_code_tokenizer_and_fim_encoding")
    run = tmp_path / "copilot"
    run.mkdir()
    (run / "tokenizer.json").write_bytes(tok_p.read_bytes())
    vocab = Tokenizer.from_file(str(tok_p)).get_vocab_size()
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=vocab, n_layer=1, n_head=2, n_embd=32, block_size=128))
    torch.save({"model": m.state_dict(), "config": m.config.__dict__, "step": 1}, run / "ckpt.pt")
    json.dump({"type": "scratch", "tok": "tokenizer.json", "task": "code"}, open(run / "meta.json", "w"))
    from chat_engine import Engine

    e = Engine(str(run))
    outs = list(e.code_complete("def add(a, b):\n    return " * 30, "\n\nprint(add(1, 2))\n" * 20, max_tokens=12))
    assert len(outs) <= 12 and all(isinstance(o, str) for o in outs)
    evs = list(e.write("code_complete", "x = <cursor>\nprint(x)", sampling={"max_tokens": 8}))
    assert evs[-1]["type"] == "final" and "balanced" in evs[-1]["check"]["checks"]


# ---------------------------------------------------------------- fine-tuning examples
def test_code_pairs_mix_fim_continuation_and_chat(code_corpus, tmp_path):
    from transformers import PreTrainedTokenizerFast

    from finetune.finetune_lora import code_pairs, encode

    tok_p = code_corpus / "bpe-code.json"
    if not tok_p.exists():
        pytest.skip("needs test_code_tokenizer_and_fim_encoding")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_p), eos_token=EOT, pad_token=EOT)
    tok.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    pairs = code_pairs(tok, str(code_corpus / "code"), 100, 256, random.Random(0), fim_rate=0.5, chat_share=0.2)
    assert len(pairs) == 100
    fims = [p for p in pairs if p[3] and p[0].startswith("<|fim_prefix|>")]
    chats = [p for p in pairs if not p[3]]
    conts = [p for p in pairs if p[3] and not p[0].startswith("<|fim_prefix|>")]
    assert fims and chats and conts
    assert chats[0][1].startswith("```") and "programming assistant" in chats[0][2]
    only_py = code_pairs(tok, str(code_corpus / "code"), 30, 256, random.Random(0), langs={"python"})
    assert all("path: p/m" in p[0] or "p/m" in p[0] for p in only_py)
    enc = encode(tok, fims[:5], 256)
    ids, labels = enc[0]
    assert labels[-1] == tok.convert_tokens_to_ids(EOT) and labels[0] == -100  # loss only on the middle + end


# ---------------------------------------------------------------- Studio
@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app
    from studio import copilot, db, jobs

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "studio.db"))
    db.init()
    monkeypatch.setattr(copilot, "CODE", str(tmp_path / "code"))
    monkeypatch.setattr(copilot, "CFG", str(tmp_path / "code" / "sources.json"))
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda kind, name, cmd, log, run=None, cleanup=(), env=None:
                        launched.append((kind, name, cmd, env)) or {"running": True})
    from studio import auth

    monkeypatch.setattr(auth, "user_for_token", lambda tok: {"id": 1, "username": "admin", "role": "admin", "prefs": {}})
    c = TestClient(app.app)
    c.launched = launched
    return c


def test_studio_code_sources_and_build(client, project, tmp_path):
    r = client.post("/api/code/config", json={"local_dirs": [str(project)], "github_repos": 5, "github_token": "ghp_x",
                                              "licenses": ["mit", "gpl-3.0"], "langs": "python;rm -rf"})
    cfg = r.json()["config"]
    assert cfg["github_token"] is None and cfg["github_token_set"] and cfg["licenses"] == ["mit"]
    assert ";" not in cfg["langs"]
    # the token stays saved when the form sends none
    assert client.post("/api/code/config", json={"local_dirs": [str(project)]}).json()["config"]["github_token_set"]
    r = client.post("/api/code/build", json={"local_dirs": [str(project)], "github_repos": 5, "what": "all"})
    assert r.status_code == 200, r.text
    kind, name, cmd, env = client.launched[-1]
    assert cmd[1] == "data/build_code.py" and "--local" in cmd and "--github-repos" in cmd
    assert env == {"GITHUB_TOKEN": "ghp_x"} and "ghp_x" not in " ".join(cmd)
    assert client.post("/api/code/build", json={"local_dirs": [str(tmp_path / "missing")], "what": "local"}).status_code == 400
    assert client.post("/api/code/build", json={"github_list": "not a repo", "what": "github"}).status_code == 400
    st = client.get("/api/code").json()
    assert st["sources"] == [] and "mit" in st["licenses"]


def test_studio_folder_picker_and_delete(client, project, tmp_path):
    r = client.get("/api/fs/dirs", params={"path": str(project)}).json()
    assert r["project"] is False and {d["name"] for d in r["dirs"]} == {"src", "node_modules", "generated"}
    assert client.get("/api/fs/dirs", params={"path": str(tmp_path / "nope")}).status_code == 404
    from studio import copilot

    d = os.path.join(copilot.CODE, "local", "x-123456")
    os.makedirs(d)
    open(os.path.join(d, "shard_00000.txt"), "w").write("# path: a.py\nx=1\n")
    assert [s["id"] for s in client.get("/api/code").json()["sources"]] == ["local/x-123456"]
    assert client.delete("/api/code/sources/..%2F..%2Fetc").status_code == 404
    assert client.delete("/api/code/sources/local/x-123456").json()["sources"] == []


def test_studio_trains_a_copilot(client, monkeypatch, tmp_path):
    from studio import copilot

    import app

    monkeypatch.setattr(app, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(app, "ROOT", str(tmp_path))  # no prepared data/bin-code from a real Studio
    r = client.post("/api/train", json={"task": "code", "mode": "lora", "name": "cp"})
    assert r.status_code == 400 and "Collect some code" in r.json()["detail"]
    os.makedirs(os.path.join(copilot.CODE, "github"))
    open(os.path.join(copilot.CODE, "github", "shard_00000.txt"), "w").write("# path: a.py\nx=1\n")
    r = client.post("/api/train", json={"task": "code", "mode": "lora", "name": "cp", "base": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                                        "fim_rate": 0.4, "code_langs": "python", "examples": 0, "instructions": True})
    assert r.status_code == 200, r.text
    cmd = client.launched[-1][2]
    assert "--code" in cmd and cmd[cmd.index("--fim-rate") + 1] == "0.4" and "--instructions" not in cmd
    assert cmd[cmd.index("--code-langs") + 1] == "python"
    r = client.post("/api/train", json={"task": "code", "mode": "scratch", "name": "cps"})
    assert r.status_code == 400 and "Copilot" in r.json()["detail"]


def test_formats_offer_code_modes():
    from writing import formats as F

    ids = {f["id"]: f for f in F.public_list()}
    assert ids["code"]["group"] == ids["code_complete"]["group"] == "Code"
    assert F.check("code_complete", "for x in y:\n    print(x)\n")["score"] == 100
    assert F.check("code", "f(((")["checks"]["balanced"] is False


def test_lora_copilot_trains_and_completes(code_corpus, tmp_path):
    """finetune_lora.py --code on a tiny random model with FIM tokens, then the Studio's completion."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok_p = code_corpus / "bpe-code.json"
    if not tok_p.exists():
        pytest.skip("needs test_code_tokenizer_and_fim_encoding")
    base = str(tmp_path / "base")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_p), eos_token=EOT, pad_token=EOT)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)).save_pretrained(base)
    tok.save_pretrained(base)
    out = str(tmp_path / "copilot")
    log = run_script("finetune/finetune_lora.py", "--base", base, "--out", out, "--max-steps", "2", "--batch", "1",
                     "--grad-accum", "1", "--seq", "128", "--eval-every", "2", "--save-every", "2", "--device", "cpu",
                     "--examples", "0", "--code", str(code_corpus / "code"), "--code-examples", "40", "--lora-r", "4")
    assert "fill-in-the-middle" in log
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["task"] == "code" and meta["fim"] is True
    from chat_engine import Engine

    e = Engine(out, backend="hf")
    outs = list(e.code_complete("def add(a, b):\n    return ", "\n", max_tokens=6))
    assert all(isinstance(o, str) for o in outs)


# ---------------------------------------------------------------- automatic updates (data/code_update.py)
@pytest.fixture
def auto_env(fake_github, tmp_path, monkeypatch):
    from data import build_code, code_update

    monkeypatch.setattr(build_code, "API", fake_github)
    monkeypatch.setattr(build_code, "CODELOAD", fake_github)
    code = tmp_path / "code"
    for k, v in {"CODE": code, "CODE_CFG": code / "sources.json", "STATE": tmp_path / "auto",
                 "SETTINGS": tmp_path / "auto" / "settings.json", "STATE_P": tmp_path / "auto" / "state.json",
                 "PENDING": tmp_path / "auto" / "pending", "LOG": tmp_path / "auto" / "log.txt",
                 "LOCAL_SEEN": tmp_path / "auto" / "local.hashes",
                 "ROOT": tmp_path}.items():
        monkeypatch.setattr(code_update, k, str(v))
    code.mkdir()
    (code / "sources.json").write_text(json.dumps({"langs": "python,javascript", "licenses": ["mit", "apache-2.0"],
                                                   "min_stars": 10, "retries": 1}))
    monkeypatch.setitem(REPOS, "erin/second", {"license": {"spdx_id": "MIT"},
                                               "files": {"LICENSE": "MIT License", "s.py": PY_FILE + "# erin\n"}})
    monkeypatch.setitem(REPOS, "alice/good", json.loads(json.dumps(REPOS["alice/good"])))  # restored afterwards
    QUERIES.clear()
    return code_update


def pending_docs(cu):
    return read_docs(cu.PENDING) if os.path.isdir(cu.PENDING) else []


def test_scheduled_updates_add_new_repos_then_only_new_commits(auto_env):
    cu = auto_env
    s = {**cu.DEFAULTS, "github_new": 1, "refresh": True, "topics": "CLI, #web, bad topic!"}
    cu.run_once(s, token="")
    st = json.load(open(os.path.join(cu.CODE, "github", "progress.json")))
    assert st["done"] == ["alice/good"] and len(pending_docs(cu)) == 2
    assert QUERIES and all(" topic:" in q for q in QUERIES) and any("topic:cli" in q for q in QUERIES)
    cu.run_once(s, token="")  # one more repository per run; alice/good has no new commits
    st = json.load(open(os.path.join(cu.CODE, "github", "progress.json")))
    assert st["done"] == ["alice/good", "erin/second"] and len(pending_docs(cu)) == 3
    # a new commit in alice/good: only the new file is added
    REPOS["alice/good"]["files"]["lib/mul.py"] = PY_FILE.replace("add", "mul")
    REPOS["alice/good"]["pushed_at"] = "2026-02-01T00:00:00Z"
    s["github_new"] = 0
    assert cu.run_once(s, token="").startswith("1 new file;")
    assert any("alice/good/lib/mul.py" in d for d in pending_docs(cu)) and len(pending_docs(cu)) == 4
    attr = [json.loads(l) for l in open(os.path.join(cu.CODE, "github", "attribution.jsonl"))]
    assert attr[-1]["repo"] == "alice/good" and attr[-1]["files"] == 3 and attr[-1]["added"] == 1
    # the licence changed to GPL: new commits are not taken any more
    REPOS["alice/good"]["license"] = {"spdx_id": "GPL-3.0"}
    REPOS["alice/good"]["files"]["lib/div.py"] = PY_FILE.replace("add", "div")
    REPOS["alice/good"]["pushed_at"] = "2026-03-01T00:00:00Z"
    assert cu.run_once(s, token="").startswith("0 new files")
    assert not any("div.py" in d for d in read_docs(cu.CODE))


def test_scheduled_updates_train_the_copilot_once_enough_is_new(auto_env, monkeypatch, tmp_path):
    cu = auto_env
    run = tmp_path / "runs" / "my-copilot"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"type": "lora", "task": "code", "base": "x"}))
    trained = []
    monkeypatch.setattr(cu, "train", lambda path, steps, mix: trained.append((path, steps, len(pending_docs(cu)))))
    s = {**cu.DEFAULTS, "github_new": 1, "train_model": "my-copilot", "train_steps": 7, "min_files": 3}
    assert "waiting" in cu.run_once(s, token="") and not trained  # 2 new files, training starts at 3
    assert "trained on 3 files" in cu.run_once(s, token="")
    assert trained == [(str(run), 7, 3)] and pending_docs(cu) == [] and cu.load_state()["trained_files"] == 3


def test_a_failed_run_is_queued_by_the_next_one(auto_env, monkeypatch, project):
    cu = auto_env
    real = cu.add_pending
    monkeypatch.setattr(cu, "add_pending", lambda docs: (_ for _ in ()).throw(RuntimeError("disk full")))
    s = {**cu.DEFAULTS, "github_new": 1}
    with pytest.raises(RuntimeError):
        cu.run_once(s, token="")
    assert json.load(open(os.path.join(cu.CODE, "github", "progress.json")))["done"] == ["alice/good"]
    monkeypatch.setattr(cu, "add_pending", real)
    s["github_new"] = 0
    cu.run_once(s, token="")  # nothing new on GitHub, but the files of the failed run are queued now
    assert len(pending_docs(cu)) == 2
    cu.run_once(s, token="")
    assert len(pending_docs(cu)) == 2
    # project folders: what was there before is not queued, a new file is
    cfg = json.load(open(cu.CODE_CFG))
    json.dump({**cfg, "local_dirs": [str(project)]}, open(cu.CODE_CFG, "w"))
    from data import build_code

    build_code.local([str(project)], cu.CODE, set(), 256)
    s.update(local=True, refresh=False)
    cu.run_once(s, token="")
    assert len(pending_docs(cu)) == 2
    (project / "src" / "new.py").write_text(PY_FILE.replace("add", "sub"))
    cu.run_once(s, token="")
    assert len(pending_docs(cu)) == 3 and any("myapp/src/new.py" in d for d in pending_docs(cu))


def test_scratch_update_mixes_earlier_code_only_with_the_same_tokenizer(auto_env, tmp_path):
    cu = auto_env
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "bpe-code.json").write_text('{"model": 1}')
    (tmp_path / "run.json").write_text('{"model": 1}')
    (tmp_path / "other.json").write_text('{"model": 2}')
    assert cu.same_tokenizer(str(tmp_path / "run.json")) and not cu.same_tokenizer(str(tmp_path / "other.json"))


def test_lora_update_command_trains_only_the_new_code(auto_env, tmp_path):
    cu = auto_env
    run = tmp_path / "runs" / "cp"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(json.dumps({"step": 40}) + "\n")
    cmd = cu.lora_cmd(str(run), {"base": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}, 10, str(tmp_path / "empty"))
    assert cmd[cmd.index("--code") + 1] == cu.PENDING and cmd[cmd.index("--examples") + 1] == "0"
    assert cmd[cmd.index("--init-adapter") + 1] == str(run / "adapter") and cmd[cmd.index("--step-offset") + 1] == "40"
    assert cmd[cmd.index("--max-steps") + 1] == "10"
    with pytest.raises(SystemExit, match="not a copilot"):
        (run / "meta.json").write_text(json.dumps({"type": "lora", "base": "x"}))
        cu.train(str(run), 5, 1.0)


def test_os_schedule_keeps_other_jobs(monkeypatch, tmp_path):
    from core import schedule

    crontab = {"text": "0 */6 * * * cd /x && python data/crawl.py >> log 2>&1  # TextLLM news crawler\n"}

    class R:
        def __init__(self, out=""):
            self.returncode, self.stdout, self.stderr = 0, out, ""

    def run(args, input=None, **kw):
        if args == ["crontab", "-l"]:
            return R(crontab["text"])
        crontab["text"] = input
        return R()

    monkeypatch.setattr(schedule.os, "name", "posix")
    monkeypatch.setattr(schedule.subprocess, "run", run)
    schedule.install("TextLLM code updates", ["python", "data/code_update.py"], str(tmp_path / "log.txt"), 24, "/x")
    lines = crontab["text"].splitlines()
    assert len(lines) == 2 and lines[1].startswith("0 3 * * * cd /x && python data/code_update.py")
    assert schedule.status("TextLLM code updates")["installed"] and schedule.status("TextLLM news crawler")["installed"]
    schedule.remove("TextLLM code updates")
    assert crontab["text"].splitlines() == lines[:1]


def test_studio_automatic_updates(client, monkeypatch, tmp_path):
    from data import code_update as cu
    from studio import copilot, jobs

    for k in ("STATE", "SETTINGS", "STATE_P", "LOG"):
        monkeypatch.setattr(cu, k, str(tmp_path / "auto" / os.path.basename(getattr(cu, k))))
    monkeypatch.setattr(jobs, "runs", lambda: [{"name": "cp", "task": "code"}, {"name": "writer", "task": None}])
    monkeypatch.setattr(cu, "schedule_status", lambda: {"installed": False, "kind": "cron", "detail": ""})
    a = client.get("/api/code").json()["auto"]
    assert a["settings"]["enabled"] is False and a["models"] == ["cp"] and a["next"] is None
    r = client.post("/api/code/auto", json={"enabled": True, "every_hours": 0, "github_new": 5, "topics": "CLI, #Web, bad!",
                                            "train_model": "cp"})
    a = r.json()
    assert a["settings"]["every_hours"] == 1 and a["settings"]["topics"] == "cli,web"
    assert a["settings"]["train_model"] == "cp" and a["next"] is not None
    assert client.post("/api/code/auto", json={"train_model": "writer"}).status_code == 400
    client.post("/api/code/config", json={"github_token": "ghp_secret"})
    r = client.post("/api/code/auto/run")
    assert r.status_code == 200, r.text
    kind, name, cmd, env = client.launched[-1]
    assert cmd[1:] == ["data/code_update.py"] and "ghp_secret" not in " ".join(cmd) and env is None
    assert copilot.auto_next(cu.load_settings()) > time.time()  # waits a full interval again
    monkeypatch.setattr(cu, "install_schedule", lambda h: f"cron every {h}")
    assert client.post("/api/code/auto/schedule?install=true").json()["message"] == "cron every 1.0"
