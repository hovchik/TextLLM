"""Build the writing instruction set: posts, stories and essays from rights-cleared sources only.

    python data/build_instructions.py                       # Gutenberg stories + essays, Studio ratings
    python data/build_instructions.py --hf-sets             # + OpenAssistant oasst2 and Dolly (attribution)
    python data/build_instructions.py --my-posts posts.txt  # + your own posts (separated by --- lines,
                                                            #   or JSONL {"format": "linkedin_post", "text": ...})
    # On a free Colab/Kaggle GPU, add synthetic posts and better requests from a bigger Apache 2.0 model:
    python data/build_instructions.py --synth-model Qwen/Qwen3-8B --synth-per-format 300 \\
        --backtranslate Qwen/Qwen3-8B
    # CPU only: a quantized 1.7-4B GGUF writes ~50 posts per format in a few hours
    python data/build_instructions.py --synth-model ~/models/Qwen3-4B-Q4_K_M.gguf --synth-per-format 50

Every example: {"format", "system", "prompt", "response", "source", "license"} and goes to
data/instructions/train.jsonl and val.jsonl (5%), with counts in stats.json.

Sources and licenses:
  * Project Gutenberg books tagged "Short stories" or "Essays": public domain in the US. Each story
    or essay becomes a request ("Write a short story titled ...") plus the text (instruction
    backtranslation). With --backtranslate MODEL the request is written by a model instead.
  * Your Studio ratings: liked answers and your corrections (yours).
  * --my-posts: your own posts (yours).
  * --synth-model: posts written by an Apache 2.0 model, best of 3 by the automatic checks.
  * --hf-sets: OpenAssistant oasst2 (Apache 2.0) and Databricks Dolly 15k (CC BY-SA 3.0, keep
    attribution if you publish the model), capped at 12% of the set.
Sets scraped from Reddit or social platforms are deliberately not used.
"""
import argparse, csv, hashlib, io, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
from writing import formats as F  # noqa: E402
from core import progress  # noqa: E402

OUT = os.path.join(ROOT, "data", "instructions")
PD = "public domain (US)"
POST_FORMATS = [k for k, f in F.formats().items() if f["group"] == "Social post"]


def say(m):
    print(m, flush=True)


def example(fmt, request, response, source, license_, fields=None):
    return {"format": fmt, "system": F.system_prompt(fmt, fields), "prompt": F.user_prompt(fmt, request, fields),
            "response": response.strip(), "source": source, "license": license_}


# ---------------------------------------------------------------- Gutenberg stories and essays
HEADING = re.compile(r"^(?:[IVXLC]+\.?\s+)?[A-Z][A-Z0-9 ,;:'\"!?.\-—]{2,70}$")
SKIP_TITLES = re.compile(r"^(CONTENTS|PREFACE|INTRODUCTION|INDEX|NOTES?|APPENDIX|FOOTNOTES|THE END|FINIS|"
                         r"CHAPTER [IVXLC\d]+|BOOK [IVXLC\d]+|ILLUSTRATIONS?)\.?$")


def split_pieces(text, lo, hi):
    """Split a collection on all-caps heading lines into (title, body) pieces of lo..hi words."""
    lines = text.split("\n")
    pieces, title, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip()
        n = len(body.split())
        if title and lo <= n <= hi and not SKIP_TITLES.match(title.upper()):
            pieces.append((title, re.sub(r"\n{3,}", "\n\n", body)))

    for i, l in enumerate(lines):
        s = l.strip()
        blank_around = (i == 0 or not lines[i - 1].strip()) and (i + 1 >= len(lines) or not lines[i + 1].strip())
        if s and blank_around and HEADING.match(s) and len(s.split()) <= 10:
            flush()
            title, buf = s.strip(" .").title(), []
        else:
            buf.append(l)
    flush()
    return pieces


def unwrap(body):
    """Gutenberg hard-wraps lines at ~70 chars; join them back into paragraphs."""
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    return "\n\n".join(p for p in paras if p)


def gutenberg_pieces(kind, max_books, max_pieces, workers, rng):
    import requests
    from build_corpus import CATALOG, fetch_book

    subject = {"story": "short stories", "essay": "essays"}[kind]
    cat = requests.get(CATALOG, timeout=60).text
    ids = [r["Text#"] for r in csv.DictReader(io.StringIO(cat))
           if r["Type"] == "Text" and r["Language"] == "en" and subject in (r.get("Subjects") or "").lower()
           and "poetry" not in (r.get("Subjects") or "").lower()]
    rng.shuffle(ids)
    ids = ids[:max_books]
    say(f"Gutenberg {kind}: {len(ids)} books tagged '{subject}'")
    task = progress.board().task(f"Gutenberg {kind} books", f"Gutenberg {kind} books", len(ids), "books")
    task.start(total=len(ids))
    lo, hi = F.get(kind)["words"]
    out, s = [], requests.Session()
    with ThreadPoolExecutor(workers) as pool:
        for bid, text in zip(ids, pool.map(lambda i: fetch_book(i, s), ids)):
            task.add(1)
            if not text:
                continue
            for title, body in split_pieces(text, lo, int(hi * 1.3)):
                out.append((bid, title, unwrap(body)))
            task.set(note=f"{min(len(out), max_pieces):,} of {max_pieces:,} pieces")
            if len(out) >= max_pieces:
                break
    task.finish("done", f"{min(len(out), max_pieces):,} pieces")
    return out[:max_pieces]


def pd_examples(kind, pieces, rng, bt=None):
    exs = []
    for bid, title, body in pieces:
        n = len(body.split())
        fields = {"length": str(int(round(n / 50.0) * 50) or 50)}
        if bt:
            req = bt(kind, body)
        elif kind == "story":
            req = rng.choice(["Write a short story titled “{t}”.", "Write a story called “{t}”.",
                              "Tell a short story with the title “{t}”."]).format(t=title)
        else:
            req = rng.choice(["Write an essay on “{t}”.", "Write a reflective essay titled “{t}”.",
                              "Write an essay about {t_l}."]).format(t=title, t_l=title.lower())
        exs.append(example(kind, req, body, f"gutenberg:{bid}", PD, fields))
    return exs


# ---------------------------------------------------------------- your own posts
def my_posts(path, bt=None):
    exs = []
    raw = open(path, encoding="utf-8").read()
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    else:
        # Posts separated by a line with --- (paragraphs inside a post are kept together).
        parts = re.split(r"\n\s*---+\s*\n", raw) if re.search(r"^\s*---+\s*$", raw, re.M) else \
            re.split(r"\n\s*\n", raw)
        rows = []
        for p in (x.strip() for x in parts):
            if len(p) <= 30:
                continue
            fmt = "x_post" if len(p) <= 280 else "linkedin_post" if len(p) > 600 or "\n\n" in p else "facebook_post"
            rows.append({"format": fmt, "text": p})
    for r in rows:
        fmt = r.get("format") if r.get("format") in F.formats() else "facebook_post"
        req = bt(fmt, r["text"]) if bt else r.get("request") or \
            "Write a post about: " + " ".join(re.sub(r"#\w+", "", r["text"]).split()[:7]).rstrip(",.;:") + "…"
        exs.append(example(fmt, req, r["text"], "my-posts", "own"))
    return exs


# ---------------------------------------------------------------- model helpers (GPU recommended)
def load_writer(name):
    """A generate(msgs, max_new, temp) function for an HF model id or a .gguf file (faster on CPU)."""
    from writing import hf

    if name.endswith(".gguf"):
        from llama_cpp import Llama

        llm = Llama(model_path=name, n_ctx=4096, n_threads=hf.physical_cores(), verbose=False)

        def gen_gguf(msgs, max_new=400, temp=0.9):
            r = llm.create_chat_completion(messages=msgs, max_tokens=max_new, temperature=temp, top_p=0.95)
            return r["choices"][0]["message"]["content"].strip()

        return gen_gguf
    import torch

    tok = hf.load_tokenizer(name)
    import devices

    dev = devices.pick()
    model = hf.load_causal_lm(name, devices.infer_dtype(dev)).to(dev).eval()

    def gen(msgs, max_new=400, temp=0.9):
        ids = tok(hf.chat_text(tok, msgs), return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return gen


def backtranslator(gen):
    def bt(fmt, text):
        msg = (f"Here is a {F.get(fmt)['label'].lower()}:\n\n{text[:2500]}\n\n"
               "Write the one-sentence request someone could have given a writer to get this text. "
               "Mention the topic and anything specific, but not the exact wording. Answer with the request only.")
        return gen([{"role": "user", "content": msg}], 80, 0.7).split("\n")[0].strip('" ')
    return bt


def synth_posts(gen, topics, per_format, rng):
    exs, tones = [], ["friendly", "confident", "playful", "reflective", "practical", "warm", "candid", "upbeat"]
    for fmt in POST_FORMATS:
        for i in range(per_format):
            topic, fields = rng.choice(topics), {"tone": rng.choice(tones)}
            msgs = F.messages(fmt, f"Write about {topic}", fields)
            cands = [F.clean(gen(msgs, F.get(fmt)["sampling"]["max_tokens"] + 60)) for _ in range(3)]
            scored = sorted(((F.check(fmt, c, fields)["score"], c) for c in cands), reverse=True)
            if scored[0][0] >= 100:
                exs.append(example(fmt, f"Write about {topic}", scored[0][1], "synthetic", "Apache 2.0 model output",
                                   fields))
            if i % 20 == 0:
                say(f"synthetic {fmt}: {i}/{per_format}")
    return exs


# ---------------------------------------------------------------- open instruction sets
def hf_sets(limit, rng):
    from datasets import load_dataset

    exs = []
    say("loading Dolly 15k (CC BY-SA 3.0) ...")
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        if r["category"] in ("creative_writing", "brainstorming", "general_qa") and len(r["response"]) > 200:
            req = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
            exs.append(example("free", req, r["response"], "dolly-15k", "CC BY-SA 3.0 (Databricks)"))
    say("loading OpenAssistant oasst2 (Apache 2.0) ...")
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    prompts = {r["message_id"]: r["text"] for r in ds if r["role"] == "prompter" and r["parent_id"] is None
               and r["lang"] == "en"}
    for r in ds:
        if r["role"] == "assistant" and r["parent_id"] in prompts and r.get("rank") == 0 and len(r["text"]) > 200:
            exs.append(example("free", prompts[r["parent_id"]], r["text"], "oasst2", "Apache 2.0"))
    rng.shuffle(exs)
    return exs[:limit]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stories", type=int, default=1200, help="max Gutenberg stories")
    ap.add_argument("--essays", type=int, default=1200, help="max Gutenberg essays")
    ap.add_argument("--books", type=int, default=300, help="max Gutenberg collections to read per kind")
    ap.add_argument("--no-gutenberg", action="store_true")
    ap.add_argument("--from-studio", action="store_true", help="include liked/corrected chat answers")
    ap.add_argument("--my-posts", action="append", default=[], help="txt (blank-line separated) or JSONL")
    ap.add_argument("--hf-sets", action="store_true", help="add oasst2 + Dolly (10-15%% of the mix)")
    ap.add_argument("--synth-model", help="Apache 2.0 model that writes example posts (use a GPU)")
    ap.add_argument("--synth-per-format", type=int, default=200)
    ap.add_argument("--topics", default=os.path.join(ROOT, "writing", "topics.txt"))
    ap.add_argument("--backtranslate", help="model that writes the request for each text (use a GPU)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gen_bt = load_writer(a.backtranslate) if a.backtranslate else None
    bt = backtranslator(gen_bt) if gen_bt else None
    exs = []
    if not a.no_gutenberg:
        try:
            for kind, n in (("story", a.stories), ("essay", a.essays)):
                if n > 0:
                    exs += pd_examples(kind, gutenberg_pieces(kind, a.books, n, a.workers, rng), rng, bt)
        except Exception as e:  # offline or the catalog moved: keep going with the other sources
            say(f"Gutenberg skipped: {e}")
    for p in a.my_posts:
        exs += my_posts(p, bt)
    if a.from_studio:
        from studio import feedback

        tmp = os.path.join(a.out, "_studio.jsonl")
        os.makedirs(a.out, exist_ok=True)
        n = feedback.export_liked(tmp)
        # code answers (Chat → Code formats) belong to copilot models, not to the writing instructions
        exs += [r for r in (json.loads(l) for l in open(tmp, encoding="utf-8") if l.strip())
                if F.get(r.get("format")).get("group") != "Code"]
        os.remove(tmp)
        say(f"Studio ratings: {n} liked or corrected answers")
    if a.synth_model:
        gen = gen_bt if a.synth_model == a.backtranslate else load_writer(a.synth_model)
        topics = [t.strip() for t in open(a.topics, encoding="utf-8") if t.strip()]
        exs += synth_posts(gen, topics, a.synth_per_format, rng)
    if a.hf_sets:
        try:
            exs += hf_sets(max(50, int(len(exs) * 0.14)), rng)  # ≈12% of the final set
        except Exception as e:
            say(f"open instruction sets skipped: {e}")
    seen, uniq = set(), []
    for e in exs:  # drop duplicates and empty answers
        h = hashlib.sha1(e["response"].lower().encode()).hexdigest()
        if e["response"] and h not in seen:
            seen.add(h)
            uniq.append(e)
    if not uniq:
        sys.exit("No examples were built. Check your internet connection, or pass --my-posts / --from-studio.")
    rng.shuffle(uniq)
    n_val = max(1, len(uniq) // 20)
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("val.jsonl", uniq[:n_val]), ("train.jsonl", uniq[n_val:])):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            for e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {"total": len(uniq), "train": len(uniq) - n_val, "val": n_val, "by_format": {}, "by_source": {},
             "by_license": {}}
    for e in uniq:
        src = e["source"].split(":")[0]
        for k, v in (("by_format", e["format"]), ("by_source", src), ("by_license", e["license"])):
            stats[k][v] = stats[k].get(v, 0) + 1
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    say(f"✅ {stats['train']} train / {stats['val']} val examples in {os.path.relpath(a.out, ROOT)}")
    say("formats: " + ", ".join(f"{k} {v}" for k, v in stats["by_format"].items()))
    say("licenses: " + ", ".join(f"{k} {v}" for k, v in stats["by_license"].items()))


if __name__ == "__main__":
    main()
