"""Web app: prepare data, start/stop training, watch live stats, chat with your models.

    pip install gradio pandas
    python app.py            # then open http://127.0.0.1:7860
"""
import glob, json, math, os, signal, subprocess, sys, time

import gradio as gr
import pandas as pd

from chat_engine import RUNS, ROOT, Engine, list_models

PY = sys.executable
JOB = {"proc": None, "name": None, "kind": None, "log": None}
os.makedirs(RUNS, exist_ok=True)


# ---------------------------------------------------------------- jobs
def _launch(kind, name, cmd, log_path):
    if JOB["proc"] and JOB["proc"].poll() is None:
        return f"⚠️ A job is already running ({JOB['kind']}: {JOB['name']}). Stop it first."
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "a")
    logf.write(f"\n$ {' '.join(cmd)}\n")
    logf.flush()
    JOB.update(proc=subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"}),
               name=name, kind=kind, log=log_path)
    return f"▶️ Started {kind}: {name}"


def stop_job():
    p = JOB["proc"]
    if not p or p.poll() is not None:
        return "Nothing is running."
    p.send_signal(signal.SIGINT)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()
    return f"⏹️ Stopped {JOB['name']}. The last checkpoint is kept, so tick 'Resume' to continue."


def tail(path, n=12):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 20000))
        lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------- data tab
def run_data(step, n_books, web_gb, vocab):
    log = os.path.join(RUNS, "_data", "log.txt")
    cmds = {
        "1. Download books + web text": [PY, "data/build_corpus.py", "--gutenberg-max", str(int(n_books)),
                                         "--fineweb-gb", str(web_gb)],
        "2. Train tokenizer": [PY, "tokenizer/train_tokenizer.py", "--vocab", str(int(vocab))],
        "3. Encode corpus to tokens": [PY, "tokenizer/train_tokenizer.py", "--encode"],
    }
    return _launch("data", step, cmds[step], log)


def data_status():
    raw = glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True)
    mb = sum(os.path.getsize(f) for f in raw) / 1e6
    tok = os.path.exists(os.path.join(ROOT, "tokenizer/bpe.json"))
    tb = os.path.join(ROOT, "data/bin/train.bin")
    toks = os.path.getsize(tb) / 2e6 if os.path.exists(tb) else 0
    md = (f"| step | status |\n|---|---|\n"
          f"| Raw text | {'✅' if raw else '⬜'} {len(raw)} shards, {mb:,.0f} MB |\n"
          f"| Tokenizer | {'✅ tokenizer/bpe.json' if tok else '⬜ not trained'} |\n"
          f"| Encoded tokens | {'✅' if toks else '⬜'} {toks:,.1f}M training tokens |")
    return md, tail(os.path.join(RUNS, "_data", "log.txt"))


# ---------------------------------------------------------------- train tab
def start_train(mode, name, resume, preset, vocab, max_steps, batch, accum, lr, base, examples, seq):
    name = (name or "run").strip().replace(" ", "_")
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    if mode == "From scratch":
        if not os.path.exists(os.path.join(ROOT, "data/bin/train.bin")):
            return "⚠️ Prepare the data first (Data tab, steps 1–3)."
        json.dump({"type": "scratch", "tok": os.path.join(ROOT, "tokenizer/bpe.json"), "preset": preset},
                  open(os.path.join(out, "meta.json"), "w"))
        cmd = [PY, "train.py", "--out", out, "--preset", preset, "--vocab", str(int(vocab)),
               "--max-steps", str(int(max_steps)), "--batch", str(int(batch)), "--grad-accum", str(int(accum)),
               "--lr", str(lr), "--warmup", str(max(10, int(max_steps) // 40)),
               "--eval-every", str(max(10, int(max_steps) // 80))]
        if resume:
            cmd.append("--resume")
    else:
        if not glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True):
            return "⚠️ Add some text first (Data tab step 1, or the Your texts tab)."
        cmd = [PY, "finetune/finetune_lora.py", "--out", out, "--base", base, "--examples", str(int(examples)),
               "--seq", str(int(seq)), "--batch", str(int(batch)), "--grad-accum", str(int(accum)),
               "--lr", str(lr), "--max-steps", str(int(max_steps)),
               "--eval-every", str(max(5, int(max_steps) // 20)), "--save-every", str(max(5, int(max_steps) // 10))]
    if not resume and os.path.exists(os.path.join(out, "metrics.jsonl")):
        os.rename(os.path.join(out, "metrics.jsonl"), os.path.join(out, f"metrics.{int(time.time())}.jsonl"))
    return _launch("training", name, cmd, os.path.join(out, "log.txt"))


def level(val_loss, scratch):
    """Rough, human-readable quality level from validation loss."""
    if val_loss is None:
        return "—"
    if not scratch:
        return "fine-tuning (lower is better; the base model already writes well)"
    for th, lab in [(6.5, "1 · random letters"), (5.0, "2 · real words"), (4.0, "3 · short phrases"),
                    (3.4, "4 · grammatical sentences"), (3.0, "5 · fluent paragraphs")]:
        if val_loss > th:
            return lab
    return "6 · stays on topic for a while"


def runs_list():
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(RUNS, "*"))
                  if os.path.exists(os.path.join(d, "metrics.jsonl")))


def train_stats(run):
    empty = pd.DataFrame({"step": pd.Series(dtype=int), "loss": pd.Series(dtype=float),
                          "series": pd.Series(dtype=str)})
    if not run:
        return "No run selected.", empty, ""
    d = os.path.join(RUNS, run)
    rows = []
    try:
        rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    except FileNotFoundError:
        pass
    running = JOB["proc"] is not None and JOB["proc"].poll() is None and JOB["name"] == run
    if not rows:
        return ("⏳ Starting… (loading data and model)" if running else "No stats yet."), empty, tail(os.path.join(d, "log.txt"))
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    scratch = meta.get("type", "scratch") == "scratch"
    step = tr[-1]["step"] if tr else 0
    total = rows[-1]["max_steps"]
    pct = 100 * step / max(1, total)
    eta = ""
    if len(tr) >= 2 and running:
        rate = (tr[-1]["step"] - tr[0]["step"]) / max(1e-6, tr[-1]["time"] - tr[0]["time"])
        secs = (total - step) / max(rate, 1e-9)
        eta = f"{int(secs // 3600)}h {int(secs % 3600 // 60)}m"
    last_val = va[-1]["loss"] if va else None
    best_val = min(r["loss"] for r in va) if va else None
    state = "🟢 training" if running else ("✅ finished" if step >= total else "⏸️ stopped")
    bar = f"<div style='background:#8883;border-radius:6px;height:14px'><div style='width:{pct:.1f}%;" \
          f"background:#4c8bf5;height:14px;border-radius:6px'></div></div>"
    md = (f"### {run} — {state}\n{bar}\n\n"
          f"| Metric | Value |\n|---|---|\n"
          f"| Progress | step **{step:,} / {total:,}** ({pct:.1f}%) {('· ETA ' + eta) if eta else ''} |\n"
          f"| Train loss | {tr[-1]['loss']:.3f} |\n" if tr else f"### {run} — {state}\n\n| Metric | Value |\n|---|---|\n")
    if last_val is not None:
        md += f"| Validation loss | {last_val:.3f} (best {best_val:.3f}, perplexity {math.exp(min(last_val, 20)):,.1f}) |\n"
    md += f"| Level | **{level(last_val, scratch)}** |\n"
    if tr:
        md += f"| Speed | {tr[-1].get('tokens_per_sec', 0):,.0f} tokens/s · lr {tr[-1]['lr']:.2e} |\n"
    df = pd.DataFrame([{"step": r["step"], "loss": r["loss"], "series": "train" if r["kind"] == "train" else "validation"}
                       for r in rows]).sort_values("step")
    return md, df, tail(os.path.join(d, "log.txt"))


# ---------------------------------------------------------------- chat tab
ENGINES = {}


def chat_fn(message, history, model_name, temperature, max_tokens):
    models = list_models()
    if model_name not in models:
        yield "No trained model selected yet. Train one, then press ↻ to refresh the model list."
        return
    if model_name not in ENGINES:
        yield "Loading model…"
        ENGINES.clear()
        ENGINES[model_name] = Engine(models[model_name])
    yield from ENGINES[model_name].stream(history, message, int(max_tokens), temperature)


# ---------------------------------------------------------------- your texts tab
ADD = "Add to training data (kept for the next full training)"
LEARN = "Teach a trained model now (download → train → delete the texts)"


def run_texts(uploads, urls, paths, action, model_name, passes, delete_local):
    uploads = [u if isinstance(u, str) else u.name for u in (uploads or [])]
    urls = [u.strip() for u in (urls or "").splitlines() if u.strip()]
    paths = [p.strip().strip('"') for p in (paths or "").splitlines() if p.strip()]
    if not (uploads or urls or paths):
        return "⚠️ Upload files, paste links, or give a file or folder path."
    if action == LEARN:
        models = list_models()
        if model_name not in models:
            return "⚠️ Pick a trained model to teach. Train one first, then press ↻."
        run = models[model_name]
        cmd = [PY, "learn.py", "--run", run, "--passes", str(passes)]
        log, name = os.path.join(run, "log.txt"), os.path.basename(run)
    else:
        cmd = [PY, "data/ingest.py", "--out", "data/raw/custom"]
        log, name = os.path.join(RUNS, "_data", "log.txt"), "add texts"
    if paths and delete_local:
        cmd.append("--delete-originals")
    if uploads or paths:
        cmd += ["--files", *uploads, *paths]
    if urls:
        cmd += ["--urls", *urls]
    msg = _launch("texts", name, cmd, log)
    if msg.startswith("▶️"):
        # Uploaded copies are always removed when the job ends; your own local files only if you tick the box.
        PENDING_CLEANUP.extend(uploads)
    return msg


PENDING_CLEANUP = []


def texts_log():
    p = JOB["proc"]
    if p and p.poll() is not None and PENDING_CLEANUP:
        for f in PENDING_CLEANUP:
            try:
                os.remove(f)
            except OSError:
                pass
        PENDING_CLEANUP.clear()
    return tail(JOB["log"]) if JOB["kind"] == "texts" else ""


# ---------------------------------------------------------------- UI
with gr.Blocks(title="Text LLM Studio") as demo:
    gr.Markdown("# Text LLM Studio\nPrepare data → train → watch progress → chat.")

    with gr.Tab("1 · Data"):
        with gr.Row():
            n_books = gr.Number(3000, label="Gutenberg books", precision=0)
            web_gb = gr.Number(0.3, label="FineWeb-Edu web text (GB)")
            dvocab = gr.Number(8192, label="Tokenizer vocab (8192 for CPU, 32000 for GPU)", precision=0)
        dstep = gr.Radio(["1. Download books + web text", "2. Train tokenizer", "3. Encode corpus to tokens"],
                         value="1. Download books + web text", label="Step")
        with gr.Row():
            gr.Button("Run step", variant="primary").click(run_data, [dstep, n_books, web_gb, dvocab],
                                                           dmsg := gr.Markdown())
            gr.Button("Stop").click(stop_job, None, dmsg)
        dstat = gr.Markdown()
        dlog = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

    with gr.Tab("2 · Your texts"):
        gr.Markdown("Train on your own **PDF, Word (.doc/.docx), TXT, Markdown or HTML** files, or on **links** to books "
                    "and articles. Use texts you have the right to use (your own writing, public-domain books, "
                    "openly licensed articles).")
        with gr.Row():
            uploads = gr.File(file_count="multiple", file_types=[".pdf", ".doc", ".docx", ".txt", ".md", ".html"],
                              label="Upload files")
            with gr.Column():
                urls = gr.Textbox(label="Links (one per line)", lines=4,
                                  placeholder="https://www.gutenberg.org/cache/epub/1342/pg1342.txt\nhttps://example.com/article")
                paths = gr.Textbox(label="Files or folders on this computer (one per line)", lines=2,
                                   placeholder="C:\\Books   or   /home/me/books")
                delete_local = gr.Checkbox(False, label="Delete these local files after extracting")
        action = gr.Radio([LEARN, ADD], value=LEARN, label="What to do")
        with gr.Row():
            learn_model = gr.Dropdown(list(list_models()), label="Model to teach", scale=4)
            gr.Button("↻", scale=0).click(lambda: gr.update(choices=list(list_models())), None, learn_model)
            passes = gr.Slider(1, 10, 3, step=1, label="Read each text this many times")
        with gr.Row():
            xmsg = gr.Markdown()
        with gr.Row():
            gr.Button("Start", variant="primary").click(
                run_texts, [uploads, urls, paths, action, learn_model, passes, delete_local], xmsg)
            gr.Button("Stop").click(stop_job, None, xmsg)
        xlog = gr.Textbox(label="Log (training progress also appears in the Stats tab)", lines=12, max_lines=12,
                          autoscroll=True)

    with gr.Tab("3 · Train"):
        mode = gr.Radio(["From scratch", "Fine-tune open model"], value="From scratch", label="Mode")
        with gr.Row():
            name = gr.Textbox("scratch-cpu", label="Run name")
            resume = gr.Checkbox(False, label="Resume (from scratch only)")
        with gr.Row():
            preset = gr.Dropdown(["tiny", "cpu", "small", "medium", "1b"], value="cpu", label="Model size (scratch)")
            vocab = gr.Number(8192, label="Vocab (must match tokenizer)", precision=0)
            base = gr.Textbox("Qwen/Qwen2.5-0.5B-Instruct", label="Base model (fine-tune)")
        with gr.Row():
            max_steps = gr.Number(20000, label="Steps", precision=0)
            batch = gr.Number(16, label="Batch", precision=0)
            accum = gr.Number(4, label="Grad accumulation", precision=0)
            lr = gr.Number(1e-3, label="Learning rate")
        with gr.Row():
            examples = gr.Number(4000, label="Book examples (fine-tune)", precision=0)
            seq = gr.Number(512, label="Tokens per example (fine-tune)", precision=0)

        def mode_defaults(m):
            if m == "From scratch":
                return "scratch-cpu", 20000, 16, 4, 1e-3
            return "finetune-qwen", 300, 2, 8, 2e-4
        mode.change(mode_defaults, mode, [name, max_steps, batch, accum, lr])
        with gr.Row():
            tmsg = gr.Markdown()
        with gr.Row():
            gr.Button("Start training", variant="primary").click(
                start_train, [mode, name, resume, preset, vocab, max_steps, batch, accum, lr, base, examples, seq], tmsg)
            gr.Button("Stop").click(stop_job, None, tmsg)

    with gr.Tab("4 · Stats"):
        run_sel = gr.Dropdown(runs_list(), label="Run", allow_custom_value=True)
        gr.Button("↻ Refresh runs", size="sm").click(lambda: gr.update(choices=runs_list()), None, run_sel)
        stats = gr.Markdown()
        plot = gr.LinePlot(x="step", y="loss", color="series", title="Loss (train vs validation)", height=320)
        tlog = gr.Textbox(label="Log", lines=10, max_lines=10, autoscroll=True)

    with gr.Tab("5 · Chat"):
        with gr.Row():
            model_sel = gr.Dropdown(list(list_models()), label="Model", scale=4)
            gr.Button("↻", scale=0).click(lambda: gr.update(choices=list(list_models())), None, model_sel)
            temp = gr.Slider(0, 1.5, 0.8, step=0.05, label="Creativity (temperature)")
            mtok = gr.Slider(32, 1024, 300, step=16, label="Max length (tokens)")
        gr.Markdown("A from-scratch model continues whatever you type. A fine-tuned model answers requests such as "
                    "*\"Write a short story about a lighthouse keeper\"* or *\"Write a LinkedIn post about…\"*.")
        gr.ChatInterface(chat_fn, additional_inputs=[model_sel, temp, mtok])

    timer = gr.Timer(5)
    timer.tick(data_status, None, [dstat, dlog])
    timer.tick(texts_log, None, xlog)
    timer.tick(train_stats, run_sel, [stats, plot, tlog])
    run_sel.change(train_stats, run_sel, [stats, plot, tlog])
    tmsg.change(lambda n: gr.update(choices=runs_list(), value=n), name, run_sel)
    demo.load(data_status, None, [dstat, dlog])

if __name__ == "__main__":
    demo.launch()
