"""How good is a tokenizer? Compression, per-language cost, round trips and speed, in one JSON.

    python tokenizer/train_tokenizer.py --check                 # tokenizer/bpe.json on data/raw
    python tokenizer/quality.py tokenizer/bpe.json data/raw     # the same, without the training script

Numbers (higher bytes per token = the model sees more text per step; lower tokens per word = cheaper):
  bytes/token   UTF-8 bytes of text per token. English books: ~4.3 with 32k, ~3.8 with 8k.
  tokens/word   "fertility": tokens per space-separated word. Near 1.3 is good for English;
                languages the tokenizer rarely saw (e.g. Armenian in an English corpus) go to 3-6.
  round trip    decode(encode(text)) == text (after NFC for tokenizers that normalize). Byte-level
                BPE never needs <unk>, so "unknown" is always 0; a failed round trip means a bug.
  vocab used    share of the vocabulary seen in the sample. Very low = vocabulary wasted on rare junk.

The corpus sample is grouped by writing system (Latin, Armenian, Cyrillic, ...), so a language that
makes up 2% of the corpus still gets its own row. Probes are fixed short texts (written for this
project) that show numbers, code and whitespace, which the corpus rows hide.
"""
import glob, json, os, random, sys, time, unicodedata

EOT = "<|endoftext|>"

# Writing systems by Unicode block. Enough to tell the languages a user is likely to train on apart.
SCRIPTS = (
    ("Armenian", ((0x0530, 0x058F), (0xFB13, 0xFB17))),
    ("Cyrillic", ((0x0400, 0x052F),)),
    ("Greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("Georgian", ((0x10A0, 0x10FF), (0x1C90, 0x1CBF))),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Devanagari", ((0x0900, 0x097F),)),
    ("CJK", ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF))),
    ("Latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))),
)
_TABLE = {}


def script_of(ch):
    o = ord(ch)
    s = _TABLE.get(o)
    if s is None:
        s = ""
        for name, ranges in SCRIPTS:
            if any(a <= o <= b for a, b in ranges):
                s = name
                break
        if len(_TABLE) < 200_000:
            _TABLE[o] = s
    return s


def classify(text, sample=4000):
    """The writing system most letters of the text use ('Latin', 'Armenian', ...; 'Other' if none)."""
    counts = {}
    step = max(1, len(text) // sample)
    for ch in text[::step]:
        if ch.isalpha():
            s = script_of(ch) or "Other"
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else "Other"


# Short fixed texts written for this project (no third-party text), one per thing worth watching.
PROBES = {
    "English prose": (
        "The harbour was quiet when Anna came down the hill. She had promised her grandfather that she "
        "would bring the letters back before the first snow, and she didn't intend to break that promise. "
        "\"Nobody reads letters any more,\" he'd said, smiling, \"but somebody should keep them.\""),
    "Armenian prose": (
        "Երեկոյան մենք նստեցինք պատշգամբում և խոսեցինք գալիք ճանապարհորդության մասին։ "
        "Լեռները հեռվում կապույտ էին, իսկ օդը՝ մաքուր ու զով։ "
        "Տատիկը պատմեց մի հին հեքիաթ, որը նա լսել էր իր մորից։"),
    "Numbers and dates": (
        "On 12 March 2024 the shop sold 1,250 books for $18,740.50, which is 15.6% more than in 2023. "
        "Call 010-555-0199 before 09:30. Order #4471829 ships in 3-5 days."),
    "Code": (
        "def mean(values):\n"
        "    \"\"\"Average of a list; 0.0 when empty.\"\"\"\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    total = sum(values)\n"
        "    return total / len(values)\n\n\n"
        "for i in range(10):\n"
        "    print(f\"{i:>3} -> {mean(list(range(i))):.2f}\")\n"),
    "Lists and line breaks": (
        "Shopping list:\n\n- 2 loaves of bread\n- milk\n- APPLES (green!)\n\n\n"
        "IT'S DONE. WE'LL GO TOMORROW.\n\tIndented line\n    four spaces\n"),
}


def describe(tok_path):
    """{'style', 'version', 'vocab', 'normalizer', 'special'} from the tokenizer file itself, so it is
    right for any copy (a run's tokenizer.json, an imported model) without a separate record.

    version 1 / style 'gpt2': GPT-2 word splitting (every model before this change).
    version 2 / style 'modern': NFC + GPT-4/Llama-3 style splitting (digits in groups of 3, ...)."""
    try:
        j = json.load(open(tok_path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pre = j.get("pre_tokenizer") or {}
    kinds = [p.get("type") for p in pre.get("pretokenizers", [])] if pre.get("type") == "Sequence" else [pre.get("type")]
    modern = "Split" in kinds
    norm = j.get("normalizer") or {}
    return {
        "version": 2 if modern else 1,
        "style": "modern" if modern else "gpt2",
        "vocab": len((j.get("model") or {}).get("vocab") or {}) + sum(
            1 for t in j.get("added_tokens", []) if t.get("content") not in ((j.get("model") or {}).get("vocab") or {})),
        "normalizer": norm.get("type") if norm else None,
        "special": [t["content"] for t in j.get("added_tokens", []) if t.get("special")
                    and not t["content"].startswith("<|reserved")],
    }


def _shards(raw):
    files = []
    for d in [x.strip() for x in raw.split(",") if x.strip()]:
        files += sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    return files


def corpus_sample(raw, max_bytes, seed=0, per_doc=200_000):
    """Documents from the corpus, a little from every file (files shuffled), about max_bytes in all.
    Each file contributes at most max_bytes / 8, so one huge shard can't fill the sample alone."""
    files = _shards(raw)
    random.Random(seed).shuffle(files)
    docs, got = [], 0
    cap = max(per_doc, max_bytes // 8)
    for path in files:
        if got >= max_bytes:
            break
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read(min(cap, max_bytes - got))
        except OSError:
            continue
        parts = text.split(EOT)
        if len(parts) > 1 and len(text) == cap:
            parts = parts[:-1]                 # the last one was cut by the read
        for d in parts:
            d = d.strip()
            if d:
                docs.append(d[:per_doc])
                got += len(d[:per_doc].encode("utf-8"))
    return docs


def _same(tok, text, ids, nfc):
    out = tok.decode(ids, skip_special_tokens=False)
    return out == (unicodedata.normalize("NFC", text) if nfc else text)


def check(tok_path, raw="data/raw", max_mb=8.0, out=None, compare=None, log=print):
    """Measure tok_path (and, if given, a second tokenizer on the same sample). Writes `out` (JSON)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    info = describe(tok_path) or {}
    nfc = info.get("normalizer") is not None
    vocab = tok.get_vocab_size()
    unk = tok.token_to_id("<unk>")
    run = getattr(tok, "encode_batch_fast", tok.encode_batch)

    docs = corpus_sample(raw, int(max_mb * 1024 * 1024)) if raw else []
    t0 = time.perf_counter()
    encs = run(docs) if docs else []
    secs = time.perf_counter() - t0
    by = {}
    seen = set()
    bad = 0
    for d, e in zip(docs, encs):
        g = by.setdefault(classify(d), {"docs": 0, "bytes": 0, "chars": 0, "words": 0, "tokens": 0})
        g["docs"] += 1
        g["bytes"] += len(d.encode("utf-8"))
        g["chars"] += len(d)
        g["words"] += len(d.split())
        g["tokens"] += len(e.ids)
        seen.update(e.ids)
    # round trips on a few hundred documents is plenty (decoding everything is slow and adds nothing)
    for d, e in list(zip(docs, encs))[:300]:
        if not _same(tok, d, e.ids, nfc):
            bad += 1
    n_bytes = sum(g["bytes"] for g in by.values())
    n_tok = sum(g["tokens"] for g in by.values())

    def row(g):
        return {**g, "bytes_per_token": round(g["bytes"] / max(1, g["tokens"]), 3),
                "chars_per_token": round(g["chars"] / max(1, g["tokens"]), 3),
                "tokens_per_word": round(g["tokens"] / max(1, g["words"]), 3)}

    probes = {}
    for name, text in PROBES.items():
        ids = tok.encode(text).ids
        probes[name] = {"tokens": len(ids), "bytes_per_token": round(len(text.encode()) / max(1, len(ids)), 3),
                        "round_trip": _same(tok, text, ids, nfc),
                        "pieces": [tok.decode([i]) for i in ids[:24]]}
    res = {
        "tokenizer": os.path.basename(tok_path), **info, "vocab": vocab,
        "measured": time.strftime("%Y-%m-%d %H:%M"),
        "sample_mb": round(n_bytes / 1048576, 2), "docs": len(docs),
        "bytes_per_token": round(n_bytes / max(1, n_tok), 3),
        "unknown_tokens": sum(e.ids.count(unk) for e in encs) if unk is not None else 0,
        "round_trip_failures": bad, "round_trip_checked": min(300, len(docs)),
        "vocab_used": round(len(seen) / max(1, vocab), 3),
        "encode_mb_s": round(n_bytes / 1048576 / secs, 2) if secs > 0 and n_bytes else None,
        "languages": {k: row(v) for k, v in sorted(by.items(), key=lambda kv: -kv[1]["bytes"])},
        "probes": probes,
    }
    if compare and os.path.exists(compare) and os.path.abspath(compare) != os.path.abspath(tok_path):
        other = check(compare, raw, max_mb, None, None, log=lambda *a: None)
        res["previous"] = {k: other[k] for k in ("tokenizer", "style", "version", "vocab", "bytes_per_token",
                                                 "vocab_used", "encode_mb_s", "languages", "probes")}
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    log(summary(res))
    return res


def summary(res):
    """A few lines for the console and the Studio log."""
    lines = [f"tokenizer {res['tokenizer']}: v{res.get('version')} ({res.get('style')}), vocabulary {res['vocab']:,}"]
    if res["docs"]:
        lines.append(f"  corpus sample {res['sample_mb']} MB, {res['docs']:,} documents: {res['bytes_per_token']} bytes/token, "
                     f"vocab used {res['vocab_used']:.0%}, unknown {res['unknown_tokens']}, "
                     f"round-trip failures {res['round_trip_failures']}/{res['round_trip_checked']}, "
                     f"encode {res['encode_mb_s']} MB/s")
        for k, g in res["languages"].items():
            lines.append(f"  {k:<10} {g['bytes'] / 1048576:7.2f} MB  {g['bytes_per_token']:5.2f} bytes/token  "
                         f"{g['tokens_per_word']:5.2f} tokens/word")
    for k, p in res["probes"].items():
        lines.append(f"  probe {k:<22} {p['tokens']:4d} tokens  {p['bytes_per_token']:5.2f} bytes/token"
                     + ("" if p["round_trip"] else "  ROUND TRIP FAILED"))
    prev = res.get("previous")
    if prev:
        lines.append(f"  previous {prev['tokenizer']} (v{prev['version']}): {prev['bytes_per_token']} bytes/token")
    return "\n".join(lines)


def stats_path(tok_path):
    return os.path.splitext(tok_path)[0] + ".stats.json"


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tok")
    ap.add_argument("raw", nargs="?", default="data/raw")
    ap.add_argument("--sample-mb", type=float, default=8.0)
    ap.add_argument("--compare")
    ap.add_argument("--out")
    a = ap.parse_args()
    check(a.tok, a.raw, a.sample_mb, a.out, a.compare)
