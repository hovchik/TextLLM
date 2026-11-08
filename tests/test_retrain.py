"""retrain.py: a new model from a trained one on other data; the original is never changed."""
import hashlib, importlib, json, os, shutil, sys, uuid

import numpy as np
import pytest
import torch

from conftest import EOT, ROOT, run_script, sample_text


def digest(path):
    h = hashlib.sha1()
    for dp, _, fns in sorted(os.walk(path)):
        for f in sorted(fns):
            h.update(f.encode())
            h.update(open(os.path.join(dp, f), "rb").read())
    return h.hexdigest()


@pytest.fixture
def work():
    """A folder inside the project: saved training commands may only point inside it (core/commands.py)."""
    d = os.path.join(ROOT, "runs", f"_test_retrain_{uuid.uuid4().hex[:8]}")
    os.makedirs(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def write_texts(folder, seed, n=60):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "shard_0.txt"), "w", encoding="utf-8") as f:
        f.write(EOT.join(sample_text(n, seed=seed)))
    return folder


@pytest.fixture
def scratch_run(bpe, work):
    """A tiny trained from-scratch model, the way the Studio leaves it (tokenizer copy, meta, train_cmd)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(bpe)
    eot = tok.token_to_id(EOT)
    data = os.path.join(work, "bin")
    os.makedirs(data)
    ids = np.concatenate([np.array(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(sample_text(80))])
    ids.tofile(os.path.join(data, "train.bin"))
    ids[: len(ids) // 4].tofile(os.path.join(data, "val.bin"))
    src = os.path.join(work, "writer")
    run_script("train.py", "--data", data, "--out", src, "--preset", "tiny", "--vocab", str(tok.get_vocab_size()),
               "--batch", "2", "--grad-accum", "1", "--max-steps", "4", "--warmup", "1", "--eval-every", "2",
               "--eval-iters", "2", "--device", "cpu")
    shutil.copy(bpe, os.path.join(src, "tokenizer.json"))
    json.dump({"type": "scratch", "tok": "tokenizer.json", "preset": "tiny"}, open(os.path.join(src, "meta.json"), "w"))
    json.dump(["train.py", "--data", os.path.relpath(data, ROOT), "--out", os.path.relpath(src, ROOT)],
              open(os.path.join(src, "train_cmd.json"), "w"))
    return src


SMALL = ["--batch", "2", "--grad-accum", "1", "--device", "cpu"]


def test_scratch_retrain_makes_a_new_model_and_keeps_the_original(scratch_run, work):
    before = digest(scratch_run)
    news = write_texts(os.path.join(work, "news"), seed=5)
    out = os.path.join(work, "writer-news")
    log = run_script("retrain.py", "--from", scratch_run, "--out", out, "--source", news, "--steps", "6", *SMALL)
    assert "is unchanged" in log
    assert digest(scratch_run) == before, "the original model was changed"
    ck = torch.load(os.path.join(out, "ckpt.pt"), map_location="cpu", weights_only=False)
    assert ck["step"] == 6
    src_ck = torch.load(os.path.join(scratch_run, "ckpt.pt"), map_location="cpu", weights_only=False)
    assert ck["config"] == src_ck["config"]
    assert open(os.path.join(out, "tokenizer.json"), "rb").read() == open(os.path.join(scratch_run, "tokenizer.json"), "rb").read()
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["type"] == "scratch" and meta["preset"] == "tiny"
    assert meta["retrained"]["from"] == "writer" and meta["retrained"]["sources"] == [os.path.relpath(news, ROOT)]
    rows = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[0]["step"] <= 6 and rows[-1]["max_steps"] == 6, "the new run has its own fresh curve"
    # stopped half way? --resume runs the saved command again (here it is already finished)
    cmd = json.load(open(os.path.join(out, "train_cmd.json")))
    assert cmd[0] == "train.py" and not any(os.path.isabs(c) for c in cmd)
    run_script("retrain.py", "--resume", "--out", out)
    # the name is taken now
    with pytest.raises(AssertionError, match="already holds a model"):
        run_script("retrain.py", "--from", scratch_run, "--out", out, "--source", news, "--steps", "2", *SMALL)
    # it really starts from the original's weights: a tiny learning rate leaves them (almost) as they were
    out2 = os.path.join(work, "writer-same")
    run_script("retrain.py", "--from", scratch_run, "--out", out2, "--source", news, "--steps", "1",
               "--lr", "1e-12", "--min-lr", "1e-12", *SMALL)
    w2 = torch.load(os.path.join(out2, "ckpt.pt"), map_location="cpu", weights_only=False)["model"]
    assert all(torch.allclose(w2[k].float(), v.float(), atol=1e-5) for k, v in src_ck["model"].items())
    with pytest.raises(AssertionError, match="must be a new name"):
        run_script("retrain.py", "--from", scratch_run, "--out", scratch_run, "--source", news, *SMALL)


def test_scratch_mixes_in_old_data_only_with_the_same_tokenizer(scratch_run, work, bpe, monkeypatch, tmp_path):
    import retrain

    news = write_texts(os.path.join(work, "news"), seed=7, n=20)
    shared = str(tmp_path / "bpe.json")
    shutil.copy(bpe, shared)
    monkeypatch.setattr(retrain, "shared_tokenizer", lambda *_: shared)  # stands in for tokenizer/bpe.json

    def build(name, mix):
        a = retrain.get_args(["--from", scratch_run, "--out", os.path.join(work, name), "--source", news,
                              "--steps", "3", "--mix", str(mix), *SMALL])
        cmd, keep = retrain.prepare(a, str(tmp_path))
        assert keep and cmd[0] == "train.py"
        d = os.path.join(work, name, retrain.DATA_DIR)
        return os.path.getsize(os.path.join(d, "train.bin")) // 2

    only_new = build("a", 0)
    mixed = build("b", 1.0)
    assert mixed > only_new * 1.5, "the old training data was not mixed in"
    # a rebuilt shared tokenizer: its data/bin means other words to this model, so it is left out
    from tokenizer.train_tokenizer import train
    write_texts(str(tmp_path / "other"), seed=3, n=100)
    train(str(tmp_path / "other"), 300, shared, 0)
    assert build("c", 1.0) == only_new


def test_data_bin_needs_the_models_own_tokenizer(scratch_run, work, tmp_path):
    import retrain
    from tokenizer.train_tokenizer import train

    data = os.path.join(work, "bin")
    other = str(tmp_path / "other.json")
    write_texts(str(tmp_path / "raw"), seed=3, n=100)
    train(str(tmp_path / "raw"), 300, other, 0)
    a = retrain.get_args(["--from", scratch_run, "--out", os.path.join(work, "x"), "--data-bin", data,
                          "--data-tok", other, "--steps", "2", *SMALL])
    with pytest.raises(SystemExit, match="not this model's tokenizer"):
        retrain.prepare(a, str(tmp_path))
    a = retrain.get_args(["--from", scratch_run, "--out", os.path.join(work, "y"), "--data-bin", data,
                          "--data-tok", os.path.join(scratch_run, "tokenizer.json"), "--steps", "2", *SMALL])
    cmd, keep = retrain.prepare(a, str(tmp_path))
    assert "--data" in cmd


def test_private_texts_are_not_kept_and_not_resumable(scratch_run, work, tmp_path):
    mine = tmp_path / "mine.txt"
    mine.write_text("\n\n".join(sample_text(30, seed=11)), encoding="utf-8")
    out = os.path.join(work, "writer-mine")
    run_script("retrain.py", "--from", scratch_run, "--out", out, "--files", str(mine), "--steps", "3", *SMALL)
    assert os.path.exists(os.path.join(out, "ckpt.pt"))
    assert not os.path.exists(os.path.join(out, "train_cmd.json"))
    assert not os.path.exists(os.path.join(out, "retrain-data"))
    assert mine.exists(), "the file was deleted without --delete-originals"


# ---------------------------------------------------------------- LoRA
@pytest.fixture
def tiny_base(bpe, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = str(tmp_path / "base")
    tok = PreTrainedTokenizerFast(tokenizer_file=bpe, eos_token=EOT, pad_token=EOT)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_lora_retrain_continues_the_adapter_into_a_new_run(tiny_base, work, tmp_path):
    src = os.path.join(work, "qwen-writer")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"prompt": f"Write post {i}", "response": t}) + "\n"
                             for i, t in enumerate(sample_text(12))))
    style = {"style": {"id": "me", "name": "Me", "posts": 12}}
    run_script("finetune/finetune_lora.py", "--base", tiny_base, "--out", src, "--max-steps", "2", "--batch", "1",
               "--grad-accum", "1", "--seq", "64", "--eval-every", "2", "--save-every", "2", "--device", "cpu",
               "--extra", str(pairs), "--examples", "0", "--lora-r", "4", "--meta-extra", json.dumps(style))
    before = digest(src)
    a_dir = write_texts(os.path.join(work, "custom-a"), seed=2, n=20)
    b_dir = write_texts(os.path.join(work, "custom-b"), seed=4, n=20)
    out = os.path.join(work, "qwen-writer-v2")
    run_script("retrain.py", "--from", src, "--out", out, "--source", f"{a_dir},{b_dir}", "--steps", "2",
               "--batch", "1", "--grad-accum", "1", "--seq", "128", "--min-doc-chars", "100", "--device", "cpu")
    assert digest(src) == before, "the original model was changed"
    assert os.path.isdir(os.path.join(out, "adapter"))
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["type"] == "lora" and meta["base"] == tiny_base
    assert meta["style"] == style["style"], "the writing style profile did not carry over"
    assert meta["retrained"]["from"] == "qwen-writer"
    cmd = json.load(open(os.path.join(out, "train_cmd.json")))
    assert cmd[0] == "finetune/finetune_lora.py" and "--init-adapter" in cmd
    from core import commands
    commands.check(cmd)  # Resume on the Runs page accepts it


def test_finetune_reads_several_folders(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "finetune"))
    import finetune_lora

    a = write_texts(str(tmp_path / "a"), 1, 3)
    b = write_texts(str(tmp_path / "b"), 2, 3)
    assert len(finetune_lora.txt_files(f"{a},{b}")) == 2
    assert finetune_lora.txt_files("") == []


# ---------------------------------------------------------------- Studio
def test_studio_retrain_endpoint_checks_and_launches(scratch_run, work, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app
    from studio import auth, db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "studio.db"))
    db.init()
    importlib.reload(auth)
    monkeypatch.setattr(app, "RUNS", work)
    raw = str(tmp_path / "raw")
    write_texts(os.path.join(raw, "news"), seed=1)
    monkeypatch.setattr(app, "RAW", raw)
    launched = {}
    monkeypatch.setattr(app, "_launch", lambda kind, name, cmd, log, **kw: launched.update(kind=kind, name=name, cmd=cmd) or {"ok": True})
    c = TestClient(app.app)
    c.post("/api/auth/setup", json={"username": "admin", "password": "secret123"})
    post = lambda **f: c.post("/api/retrain", data={"model": "writer", **f})
    assert post(name="writer", sources="news").status_code == 400          # same name as the original
    assert post(name="../x", sources="news").status_code == 400            # not a run name
    assert post(name="writer-v2").status_code == 400                      # no data chosen
    assert post(name="writer-v2", sources="../../etc").status_code == 404  # outside data/raw
    assert c.post("/api/retrain", data={"model": "nope", "name": "x", "sources": "news"}).status_code == 404
    r = post(name="writer-v2", sources="news", steps="500", lr="0.0002")
    assert r.status_code == 200, r.text
    cmd = launched["cmd"]
    assert launched["name"] == "writer-v2" and cmd[1] == "retrain.py"
    assert cmd[cmd.index("--from") + 1] == scratch_run and cmd[cmd.index("--out") + 1] == os.path.join(work, "writer-v2")
    assert cmd[cmd.index("--source") + 1] == os.path.join(raw, "news")
    assert cmd[cmd.index("--steps") + 1] == "500" and "--gguf" not in cmd
    # a model already in that folder is never overwritten from the browser
    shutil.copy(os.path.join(scratch_run, "meta.json"), os.path.join(work, "writer-v2", "meta.json"))
    assert post(name="writer-v2", sources="news").status_code == 409
