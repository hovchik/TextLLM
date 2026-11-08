"""Fine-tuned (Hugging Face) models, with a tiny random model saved locally (no downloads)."""
import json, os, sys, threading, time
from types import SimpleNamespace

import pytest
import torch

from conftest import ROOT, run_script, sample_text


@pytest.fixture(scope="module")
def tiny_base(bpe, tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = str(tmp_path_factory.mktemp("base"))
    tok = PreTrainedTokenizerFast(tokenizer_file=bpe, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def hf_engine(base, model=None):
    """An Engine for the 'hf' backend without a run folder."""
    from chat_engine import Engine
    from writing import hf

    e = object.__new__(Engine)
    e.backend, e.device, e.lock = "hf", "cpu", threading.Lock()
    e.tok = hf.load_tokenizer(base)
    e.model = model or hf.load_causal_lm(base).eval()
    return e


SAMPLING = {"max_tokens": 40, "temperature": 0.8, "top_p": 0.95}
MSGS = [{"role": "user", "content": "Write about the harbour."}]


def test_hf_stream_produces_text(tiny_base):
    outs = list(hf_engine(tiny_base).stream_messages(MSGS, SAMPLING))
    assert outs and all(len(a) <= len(b) for a, b in zip(outs, outs[1:]))


def test_hf_stream_passes_anti_repetition_settings(tiny_base):
    seen = {}
    eng = hf_engine(tiny_base)
    real = eng.model.generate

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    eng.model.generate = spy
    s = {**SAMPLING, "repetition_penalty": 1.3, "frequency_penalty": 0.5, "no_repeat_ngram": 3}
    assert list(eng.stream_messages(MSGS, s))
    assert seen["repetition_penalty"] == 1.3 and seen["no_repeat_ngram_size"] == 3 and seen["logits_processor"]


def test_hf_stream_reports_a_failing_generate_instead_of_hanging(tiny_base):
    class Broken:
        def generate(self, **kw):
            raise RuntimeError("CUDA out of memory (simulated)")

    eng = hf_engine(tiny_base, Broken())
    box = {}

    def consume():
        try:
            list(eng.stream_messages(MSGS, SAMPLING))
        except RuntimeError as e:
            box["err"] = str(e)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(20)
    assert not t.is_alive(), "the chat stream hung after generate() failed"
    assert "out of memory" in box.get("err", "")


def test_hf_stream_stops_generation_when_the_reader_leaves(tiny_base):
    eng = hf_engine(tiny_base)
    done = threading.Event()
    real = eng.model.generate

    def generate(**kw):
        try:
            return real(**{**kw, "max_new_tokens": 5000, "min_new_tokens": 5000})
        finally:
            done.set()

    eng.model.generate = generate
    g = eng.stream_messages(MSGS, SAMPLING)
    next(g)
    t0 = time.time()
    g.close()  # the browser stopped the stream
    assert done.is_set(), "generate() was still running after the stream closed"
    assert time.time() - t0 < 20


def test_dtype_kwarg_matches_transformers_version(monkeypatch):
    import transformers

    from writing import hf

    monkeypatch.setattr(transformers, "__version__", "4.51.3")
    assert hf.dtype_kwarg() == "torch_dtype"
    monkeypatch.setattr(transformers, "__version__", "4.56.0.dev0")
    assert hf.dtype_kwarg() == "dtype"
    monkeypatch.setattr(transformers, "__version__", "5.2.0")
    assert hf.dtype_kwarg() == "dtype"


def finetune(base, out, *extra):
    return run_script("finetune/finetune_lora.py", "--base", base, "--out", out, "--max-steps", "2",
                      "--batch", "1", "--grad-accum", "1", "--seq", "64", "--eval-every", "2", "--save-every", "2",
                      "--device", "cpu", *extra)


def test_learning_private_texts_keeps_meta_and_resume_state(tiny_base, tmp_path, monkeypatch):
    out = str(tmp_path / "style-run")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"prompt": f"Write post {i}", "response": t}) + "\n"
                             for i, t in enumerate(sample_text(12))))
    style = {"style": {"id": "me", "name": "Me", "posts": 12}}
    finetune(tiny_base, out, "--extra", str(pairs), "--examples", "0", "--lora-r", "4",
             "--meta-extra", json.dumps(style))
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["style"] == style["style"] and meta["lora_r"] == 4
    state_before = open(os.path.join(out, "train_state.pt"), "rb").read()
    examples_before = open(os.path.join(out, "examples.pt"), "rb").read()

    # learn.py's step: the run's own adapter, temporary texts
    raw = tmp_path / "text"
    raw.mkdir()
    (raw / "t.txt").write_text("<|endoftext|>".join(t * 3 for t in sample_text(6, seed=9)))
    cache = os.path.join(ROOT, "data", "cache")
    cached_files = lambda: set(os.listdir(cache)) if os.path.isdir(cache) else set()
    cached = cached_files()
    finetune(tiny_base, out, "--raw", str(raw), "--init-adapter", os.path.join(out, "adapter"),
             "--examples", "4", "--min-doc-chars", "100", "--seq", "256", "--private-texts")
    meta2 = json.load(open(os.path.join(out, "meta.json")))
    assert meta2["style"] == style["style"], "the style profile was lost"
    assert meta2["lora_r"] == 4
    assert open(os.path.join(out, "train_state.pt"), "rb").read() == state_before
    assert open(os.path.join(out, "examples.pt"), "rb").read() == examples_before
    assert cached_files() == cached, "private texts were cached in data/cache"


def test_write_meta_for_a_new_run_starts_fresh(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "finetune"))
    import finetune_lora

    out = tmp_path / "r"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({"type": "lora", "base": "old", "style": {"id": "x"}}))
    a = SimpleNamespace(out=str(out), base="new", lora_r=16, instructions="i.jsonl", book_share=0.2,
                        resume=False, init_adapter=None, meta_extra=None)
    finetune_lora.write_meta(a)
    m = json.load(open(out / "meta.json"))
    assert m == {"type": "lora", "base": "new", "lora_r": 16, "instructions": True, "book_share": 0.2}
