"""End to end on CPU with a tiny model: train.py → resume → chat (Engine) → learn.py encoding."""
import json, os, shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import EOT, run_script, sample_text


def write_bins(bpe, d):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(bpe)
    eot = tok.token_to_id(EOT)
    ids = np.concatenate([np.array(e.ids + [eot], dtype=np.uint16) for e in tok.encode_batch(sample_text(80))])
    os.makedirs(d, exist_ok=True)
    ids.tofile(os.path.join(d, "train.bin"))
    ids[: len(ids) // 4].tofile(os.path.join(d, "val.bin"))
    return tok.get_vocab_size()


def train(data, out, steps, vocab, *extra):
    return run_script("train.py", "--data", data, "--out", out, "--preset", "tiny", "--vocab", str(vocab),
                      "--batch", "2", "--grad-accum", "1", "--max-steps", str(steps), "--warmup", "1",
                      "--eval-every", "2", "--eval-iters", "2", "--device", "cpu", *extra)


@pytest.fixture(scope="module")
def run_dir(bpe, tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    data = str(d / "bin")
    vocab = write_bins(bpe, data)
    out = str(d / "scratch-test")
    log = train(data, out, 4, vocab)
    assert "device cpu" in log
    shutil.copy(bpe, os.path.join(out, "tokenizer.json"))
    json.dump({"type": "scratch", "preset": "tiny"}, open(os.path.join(out, "meta.json"), "w"))
    return SimpleNamespace(dir=out, data=data, vocab=vocab)


def test_train_writes_checkpoint_and_metrics(run_dir):
    ck = torch.load(os.path.join(run_dir.dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["config"]["vocab_size"] == run_dir.vocab
    rows = [json.loads(l) for l in open(os.path.join(run_dir.dir, "metrics.jsonl"))]
    assert any(r["kind"] == "val" for r in rows)


def test_resume_continues_from_the_checkpoint(run_dir, tmp_path):
    out = str(tmp_path / "copy")
    shutil.copytree(run_dir.dir, out)
    log = train(run_dir.data, out, 6, run_dir.vocab, "--resume")
    assert "resumed at step 4" in log
    assert torch.load(os.path.join(out, "ckpt.pt"), map_location="cpu", weights_only=False)["step"] == 6


def test_chat_engine_streams_scratch_text(run_dir):
    from chat_engine import Engine

    eng = Engine(run_dir.dir)
    assert eng.backend == "scratch"
    outs = list(eng.stream([], "the harbour", max_new_tokens=30, temperature=0.8))
    assert outs, "no text streamed"
    assert all(o.startswith("the harbour") for o in outs)
    assert all(len(a) <= len(b) for a, b in zip(outs, outs[1:]))  # the reply only grows
    ev = list(eng.write("free", "the harbour", sampling={"max_tokens": 20}))
    assert ev[-1]["type"] == "final"


def test_learn_scratch_encodes_all_documents(run_dir, tmp_path, bpe):
    import learn
    from tokenizers import Tokenizer

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    docs = sample_text(30, seed=5)
    (text_dir / "a.txt").write_text(EOT.join(docs[:15]) + EOT, encoding="utf-8")
    (text_dir / "b.txt").write_text(EOT.join(docs[15:]), encoding="utf-8")
    a = SimpleNamespace(run=run_dir.dir, passes=1, mix=0, batch=2, grad_accum=1, lr=None)
    cmd = learn.learn_scratch(a, {"preset": "tiny"}, str(text_dir), str(tmp_path))
    got = np.fromfile(tmp_path / "bin" / "val.bin", dtype=np.uint16)
    tok = Tokenizer.from_file(bpe)
    eot = tok.token_to_id(EOT)
    want = []
    for d in docs:  # what the previous one-document-at-a-time loop produced
        want += tok.encode(d.strip()).ids + [eot]
    assert got.tolist() == want
    assert "--resume" in cmd and cmd[cmd.index("--out") + 1] == run_dir.dir


def test_scratch_chat_has_anti_repetition_defaults_and_code_turns_them_off():
    from chat_engine import repeat_settings, scratch_penalties
    from writing import formats as F

    p = scratch_penalties(F.get("story")["sampling"])
    assert p["repetition"] > 1.0 and p["no_repeat_ngram"] >= 3
    code = scratch_penalties(F.get("code")["sampling"])
    assert code["repetition"] == 1.0 and code["no_repeat_ngram"] == 0
    # the user's choice wins over the defaults
    assert repeat_settings({"no_repeat_ngram": 0, "repetition_penalty": 1.5}, "scratch")["no_repeat_ngram"] == 0
    assert scratch_penalties({"repetition_penalty": 1.5})["repetition"] == 1.5


def test_gguf_ngram_processor_bans_the_repeating_token():
    import numpy as np
    from chat_engine import _ngram_processor_gguf

    out = _ngram_processor_gguf(3)(np.array([1, 2, 3, 4, 1, 2]), np.zeros(5, dtype=np.float32))
    assert out.dtype == np.float32 and np.isneginf(out[3]) and np.isfinite(out[[0, 1, 2, 4]]).all()


@pytest.mark.parametrize("fmt,msg,prompt,shown", [
    ("story", "A lighthouse keeper finds a message in a bottle.",
     "A lighthouse keeper finds a message in a bottle\n\nOnce upon a time,", "Once upon a time,"),
    ("free", "Write a little story about a lighthouse keeper", "A lighthouse keeper\n\nOnce upon a time,", "Once upon a time,"),
    ("free", "write a little story", "Once upon a time,", "Once upon a time,"),
    ("free", "Once upon a time a fox", "Once upon a time a fox", None),       # an opening: continue it
    ("free", "Give me liberty or give me death", "Give me liberty or give me death", None),  # not a request
    ("free", "Write a Python function that adds two numbers", "Write a Python function that adds two numbers", None),
    ("story", "write about the sea", "The sea\n\nOnce upon a time,", "Once upon a time,"),
    ("essay", "Write an essay on why cities need trees", "Why cities need trees\n\n", ""),
])
def test_scratch_models_get_a_document_opening_not_a_sentence_to_finish(fmt, msg, prompt, shown):
    from writing import formats as F

    assert F.scratch_opening(fmt, msg) == (prompt, shown)


def test_scratch_story_answer_starts_with_the_opening_not_mid_sentence(run_dir):
    from chat_engine import Engine

    eng = Engine(run_dir.dir)
    seen = {}
    real = eng.model.stream

    def spy(ids, *a, **kw):
        seen["ids"] = ids[0].tolist()
        return real(ids, *a, **kw)

    eng.model.stream = spy
    evs = list(eng.write("story", "A lighthouse keeper finds a message in a bottle", {}, [], {"max_tokens": 8}))
    assert evs[-1]["type"] == "final" and evs[-1]["text"].startswith("Once upon a time,")
    assert "lighthouse" not in evs[-1]["text"]                         # the title is not part of the answer
    assert seen["ids"][0] == eng.tok.token_to_id(EOT)                  # starts a new document
