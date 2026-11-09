"""Tokenizer v2 (modern) and v1 (gpt2): splitting, special tokens, language balance, quality check,
and old tokenizers keep working."""
import json, os

import pytest

from conftest import EOT, run_script, sample_text

from tokenizer import quality
from tokenizer import train_tokenizer as tt

ARMENIAN = quality.PROBES["Armenian prose"]


def corpus(root, hy_docs=2):
    """data/raw-like folders: lots of English 'books' and one small folder of Armenian text."""
    books = root / "gutenberg"
    books.mkdir(parents=True)
    for i in range(12):
        (books / f"shard_{i:05d}.txt").write_text(EOT.join(sample_text(30, seed=i)) + EOT, encoding="utf-8")
    own = root / "custom"
    own.mkdir()
    (own / "hy.txt").write_text(EOT.join([ARMENIAN] * hy_docs) + EOT, encoding="utf-8")
    return str(root)


@pytest.fixture(scope="module")
def toks(tmp_path_factory):
    d = tmp_path_factory.mktemp("toks")
    raw = corpus(d / "raw")
    v1, v2 = str(d / "v1.json"), str(d / "v2.json")
    tt.train(raw, 600, v1, 0, style="gpt2", check_mb=0)
    tt.train(raw, 600, v2, 0, style="modern", check_mb=0)
    return raw, v1, v2


def test_modern_tokenizer_splits_like_gpt4_and_llama3(toks):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(toks[2])
    pieces = [tok.decode([i]) for i in tok.encode("Paid 1234567 in 2024. DON'T stop").ids]
    digits = [p.strip() for p in pieces if p.strip().isdigit()]
    assert digits and all(len(p) <= 3 for p in digits)          # numbers in groups of up to 3 digits
    assert "".join(digits) == "12345672024"


def test_modern_tokenizer_round_trips_any_text_and_normalizes_to_nfc(toks):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(toks[2])
    for text in (ARMENIAN, "emoji 🙂🚀 and 中文", quality.PROBES["Code"], "tabs\t\tand\n\n\nlines  "):
        assert tok.decode(tok.encode(text).ids) == text
    decomposed = "café"                                     # e + combining accent
    assert tok.decode(tok.encode(decomposed).ids) == "café"  # stored once, as NFC


def test_special_tokens_keep_eot_first_and_reserve_slots(toks):
    from tokenizers import Tokenizer

    v1, v2 = Tokenizer.from_file(toks[1]), Tokenizer.from_file(toks[2])
    assert v1.token_to_id(EOT) == 0 and v1.token_to_id("<|pad|>") is None
    assert v2.token_to_id(EOT) == 0
    for t in ("<|pad|>", "<|im_start|>", "<|im_end|>", f"<|reserved_{tt.N_SPECIAL - 5}|>"):
        assert v2.token_to_id(t) is not None
    assert v2.token_to_id("<|fim_prefix|>") is None                # writing models get no FIM tokens
    assert v2.get_vocab_size() <= 600                              # a tiny corpus runs out of merges
    code = tt.special_tokens(True)
    assert code[0] == EOT and "<|fim_middle|>" in code and len(code) == tt.N_SPECIAL


def test_describe_tells_the_versions_apart(toks):
    a, b = quality.describe(toks[1]), quality.describe(toks[2])
    assert (a["version"], a["style"], a["normalizer"]) == (1, "gpt2", None)
    assert (b["version"], b["style"], b["normalizer"]) == (2, "modern", "NFC") and 300 < b["vocab"] <= 600
    assert "<|im_start|>" in b["special"] and not any("reserved" in s for s in b["special"])


def test_sample_takes_every_source_folder_and_boosts_rare_scripts(tmp_path):
    raw = corpus(tmp_path / "raw")
    files = tt.shards(raw)
    for seed in range(5):
        picked = tt.pick_files(raw, files, 3, seed)
        assert len(picked) == 3 and any(f.endswith("hy.txt") for f in picked)
    sizes = [os.path.getsize(f) for f in files]
    reps, mix = tt.language_repeats(files, sizes, 0.3)
    assert mix["Armenian"][1] > 1 and mix["Latin"][1] == 1
    assert reps[[i for i, f in enumerate(files) if f.endswith("hy.txt")][0]] == mix["Armenian"][1]
    assert tt.language_repeats(files, sizes, 1.0)[0] == [1] * len(files)


def test_quality_check_reports_languages_and_the_previous_tokenizer(tmp_path):
    raw = corpus(tmp_path / "raw", hy_docs=6)
    out = str(tmp_path / "bpe.json")
    tt.train(raw, 600, out, 0, style="gpt2", check_mb=0)
    tt.train(raw, 600, out, 0, style="modern", check_mb=1)        # replaces it; the old one is kept as .prev
    assert os.path.exists(tt.previous_path(out))
    st = json.load(open(quality.stats_path(out), encoding="utf-8"))
    assert st["version"] == 2 and st["previous"]["version"] == 1
    assert {"Latin", "Armenian"} <= set(st["languages"])
    assert st["unknown_tokens"] == 0 and st["round_trip_failures"] == 0
    assert all(p["round_trip"] for p in st["probes"].values())
    assert st["bytes_per_token"] > 1.5 and 0 < st["vocab_used"] <= 1


def test_cli_check_and_old_tokenizers_still_encode(toks, tmp_path):
    raw, v1, _ = toks
    run_script("tokenizer/train_tokenizer.py", "--check", "--tok", v1, "--raw", raw, "--sample-mb", "1")
    assert json.load(open(quality.stats_path(v1)))["version"] == 1
    run_script("tokenizer/train_tokenizer.py", "--encode", "--tok", v1, "--raw", raw, "--bin", str(tmp_path / "bin"))
    assert os.path.getsize(tmp_path / "bin" / "train.bin") > 1000


def test_studio_shows_the_tokenizer_version_and_quality(toks, tmp_path, monkeypatch):
    import shutil

    from studio import jobs

    (tmp_path / "tokenizer").mkdir()
    shutil.copy(toks[2], tmp_path / "tokenizer" / "bpe.json")
    monkeypatch.setattr(jobs, "ROOT", str(tmp_path))
    s = jobs.data_status()
    assert s["tok_info"]["version"] == 2 and s["tok_stats"] is None
    quality.check(str(tmp_path / "tokenizer" / "bpe.json"), toks[0], 1, quality.stats_path(str(tmp_path / "tokenizer" / "bpe.json")))
    assert jobs.data_status()["tok_stats"]["languages"]
    run = tmp_path / "runs" / "old-model"
    run.mkdir(parents=True)
    shutil.copy(toks[1], run / "tokenizer.json")                 # a model from before: no version in meta.json
    (run / "meta.json").write_text(json.dumps({"type": "scratch", "tok": "tokenizer.json", "preset": "cpu"}))
    assert jobs.run_summary(str(run))["tokenizer"]["version"] == 1
