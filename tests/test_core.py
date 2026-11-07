"""The core package: safe loading, link checks, path checks and saved-command checks."""
import json, os

import pytest

from core import commands, net, paths, safeload


# ---------------------------------------------------------------- safeload
def test_safeload_reads_plain_checkpoints(tmp_path):
    import torch

    p = tmp_path / "ckpt.pt"
    torch.save({"model": {"w": torch.ones(2)}, "step": 3, "config": {"n_layer": 1}, "args": {"lr": 0.1, "x": None},
                "rng": {"torch": torch.get_rng_state(), "np": {"bit_generator": "PCG64", "state": {"state": 1}}}}, p)
    ck = safeload.load(p)
    assert ck["step"] == 3 and ck["model"]["w"].tolist() == [1.0, 1.0]


def test_safeload_refuses_pickled_code(tmp_path):
    import pickle

    p = tmp_path / "evil.pt"
    with open(p, "wb") as f:  # a plain pickle that would call os.system when unpickled
        pickle.dump({"model": __import__("os").system}, f)
    with pytest.raises(ValueError):
        safeload.load(p)


def test_safeload_covers_the_real_training_checkpoint(bpe, tmp_path):
    """train.py's checkpoint (optimizer state, numpy RNG state, args) loads with weights_only."""
    import numpy as np
    import torch

    from model.gpt import GPT, GPTConfig
    from train import Loader, make_optimizer

    d = tmp_path / "bin"
    d.mkdir()
    np.random.randint(0, 400, 5000, dtype=np.uint16).tofile(d / "train.bin")
    m = GPT(GPTConfig(vocab_size=400, n_layer=1, n_head=2, n_embd=16, block_size=8))
    opt = make_optimizer(m, 1e-3, 0.1, "cpu")
    ld = Loader(str(d / "train.bin"), 2, 8, "cpu", 0, 1)
    x, y = ld.next()
    _, loss = m(x, y)
    loss.backward()
    opt.step()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "config": {"vocab_size": 400}, "step": 1,
                "best_val": 2.0, "rng": {"train": ld.rng.bit_generator.state, "val": ld.rng.bit_generator.state,
                                         "torch": torch.get_rng_state()}, "scaler": None,
                "args": {"lr": 1e-3, "device": "cpu", "compile": False}}, tmp_path / "ckpt.pt")
    ck = safeload.load(tmp_path / "ckpt.pt")
    assert ck["rng"]["train"]["bit_generator"] == "PCG64" and ck["opt"]["state"]


def test_loader_batches_are_shifted_by_one(tmp_path):
    import numpy as np

    from train import Loader

    np.arange(0, 3000, dtype=np.uint16).tofile(tmp_path / "t.bin")
    x, y = Loader(str(tmp_path / "t.bin"), 4, 16, "cpu", 0, 1).next()
    assert x.shape == (4, 16) and (y[:, :-1] == x[:, 1:]).all() and (y[:, 0] == x[:, 0] + 1).all()


def test_sample_respects_top_k_and_top_p():
    import torch

    from model.gpt import sample

    logits = torch.full((1, 1000), -10.0)
    logits[0, 7], logits[0, 42] = 10.0, 9.0
    picks = {sample(logits, 1.0, 50, 0.95).item() for _ in range(50)}
    assert picks <= {7, 42}
    assert sample(logits, 1.0, 1, 1.0).item() == 7  # top-k 1 = greedy
    assert sample(logits, 1e-9, 50, 0.95).item() == 7  # temperature 0 = greedy


# ---------------------------------------------------------------- net
@pytest.mark.parametrize("ip,public", [("8.8.8.8", True), ("127.0.0.1", False), ("10.1.2.3", False),
                                       ("192.168.0.1", False), ("169.254.169.254", False), ("::1", False),
                                       ("::ffff:10.0.0.1", False), ("2606:4700::1111", True), ("0.0.0.0", False)])
def test_public_ip(ip, public):
    assert net.is_public_ip(ip) is public


def test_check_url_refuses_local_addresses(monkeypatch):
    monkeypatch.setattr(net, "allow_private", lambda: False)
    net._dns.clear()
    for bad in ("http://127.0.0.1:7860/api", "http://localhost/x", "http://169.254.169.254/latest",
                "http://[::1]/", "ftp://example.com/a", "http://user:pw@example.com/", "http://router.local/"):
        with pytest.raises(ValueError):
            net.check_url(bad)
    monkeypatch.setattr(net, "resolve", lambda host: ["10.0.0.5"])
    with pytest.raises(ValueError):
        net.check_url("http://nas.example.com/book.pdf")
    monkeypatch.setattr(net, "resolve", lambda host: ["93.184.216.34"])
    assert net.check_url("https://example.com/story") == "https://example.com/story"


def test_check_url_can_allow_private(monkeypatch):
    monkeypatch.setenv("TEXTLLM_ALLOW_PRIVATE_URLS", "1")
    assert net.check_url("http://192.168.1.10/share/a.txt")


def test_get_checks_every_redirect_hop(monkeypatch):
    monkeypatch.setattr(net, "allow_private", lambda: False)
    monkeypatch.setattr(net, "resolve", lambda host: ["93.184.216.34"] if host == "example.com" else ["127.0.0.1"])
    seen = []

    class R:
        def __init__(self, loc):
            self.is_redirect, self.is_permanent_redirect = bool(loc), False
            self.headers = {"Location": loc} if loc else {}
            self.status_code = 302 if loc else 200

        def close(self):
            pass

        def iter_content(self, n):
            yield b"ok"

    class S:
        def get(self, url, **kw):
            seen.append(url)
            assert kw["allow_redirects"] is False
            return R("http://127.0.0.1:7860/api/users" if len(seen) == 1 else None)

    with pytest.raises(ValueError, match="local-network"):
        net.get("http://example.com/a", session=S())
    assert seen == ["http://example.com/a"]  # the redirect to the Studio itself was never followed


def test_get_caps_size(monkeypatch):
    monkeypatch.setattr(net, "check_url", lambda url: url)

    class R:
        is_redirect = is_permanent_redirect = False
        headers = {"Content-Length": "999999999"}

        def close(self):
            pass

    class S:
        def get(self, url, **kw):
            return R()

    with pytest.raises(ValueError, match="larger than"):
        net.get("http://example.com/big.pdf", session=S(), max_bytes=1000)


# ---------------------------------------------------------------- paths
def test_under_and_safe_name(tmp_path):
    base = str(tmp_path)
    assert paths.under(base, "my-run") == os.path.join(base, "my-run")
    for bad in ("..", "../x", "/etc/passwd", "C:\\x", "", "a/../../b", "\\\\server\\share"):
        assert paths.under(base, bad) is None, bad
    assert "/" not in paths.safe_name("../../evil") and not paths.safe_name("../../evil").startswith(".")
    assert paths.safe_name("  ") == "item" and paths.safe_name("..") == "item"
    assert paths.safe_name("my model v2.gguf") == "my-model-v2.gguf"
    assert paths.inside(base, os.path.join(base, "a", "b")) and not paths.inside(base, os.path.dirname(base))


# ---------------------------------------------------------------- commands
def test_commands_accept_the_studios_own_commands():
    ok = ["train.py", "--out", "runs/my-model", "--preset", "cpu", "--vocab", "8192", "--max-steps", "20000"]
    assert commands.check(ok) == ok
    assert commands.check(["finetune/finetune_lora.py", "--base", "Qwen/Qwen3-0.6B", "--out", "runs/x", "--gguf",
                           "--instructions", "data/instructions/train.jsonl"])
    # what the Studio itself writes for style and copilot runs
    assert commands.check(["finetune/finetune_lora.py", "--out", "runs/style-x", "--extra", "runs/style-x/pairs.jsonl",
                           "--meta-extra", json.dumps({"style": {"id": "x", "name": "Ann O'Neil", "posts": 3}})])
    assert commands.check(["train.py", "--out", "runs/code", "--code-langs", "python,c#,c++", "--data", "data/bin-code"])


@pytest.mark.parametrize("cmd", [
    ["evil.py", "--out", "runs/x"],                           # not a training script
    ["train.py", "--out", "../../home/me/.ssh"],              # path outside the project
    ["train.py", "--data", "/etc"],                           # absolute path outside
    ["train.py", "--out", "runs/x", "--lr", "1e-3\n--out\n/tmp"],  # control characters
    ["train.py", 5],                                          # not strings
    [],
    "train.py --out runs/x",
])
def test_commands_refuse_odd_commands(cmd):
    with pytest.raises(ValueError):
        commands.check(cmd)


def test_import_bundle_drops_a_bad_saved_command(tmp_path, monkeypatch):
    """A model bundle from another PC whose train_cmd.json names a different script is imported
    (the weights are fine) but the command is not kept, so Resume can't run it."""
    import zipfile

    import transfer

    monkeypatch.setattr(transfer, "RUNS", str(tmp_path / "runs"))
    z = tmp_path / "evil.textllm.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("textllm.json", json.dumps({"format": 1, "name": "evil", "type": "scratch"}))
        zf.writestr("run/meta.json", json.dumps({"type": "scratch", "tok": "/abs/path"}))
        zf.writestr("run/tokenizer.json", "{}")
        zf.writestr("run/train_cmd.json", json.dumps(["../../../usr/bin/python3", "-c", "print(1)"]))
    notes = []
    transfer.import_bundle(str(z), log=notes.append)
    assert not os.path.exists(tmp_path / "runs" / "evil" / "train_cmd.json")
    assert any("dropped" in n for n in notes)
    assert transfer.resume_cmd(str(tmp_path / "runs" / "evil"))[1] == "train.py"


def test_check_url_without_dns_is_offline_safe(monkeypatch):
    monkeypatch.setattr(net, "allow_private", lambda: False)
    monkeypatch.setattr(net, "resolve", lambda host: pytest.fail("no lookup expected"))
    assert net.check_url("https://news.example.org/feed", dns=False)
    for bad in ("http://10.0.0.1/", "http://[::1]/", "http://localhost/", "ftp://x.org/"):
        with pytest.raises(ValueError):
            net.check_url(bad, dns=False)
