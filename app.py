"""Text LLM Studio: web app for data, training, analytics, chat and feedback.

    pip install -r requirements.txt
    python app.py                 # then open http://127.0.0.1:7860
    python app.py --port 8000 --host 0.0.0.0

Sign-in is required. The first visit creates the admin account; admins add more people on the
Users page (or with `python -m studio.auth add NAME`). Regular users only see Chat and Settings.
"""
import argparse, glob, json, os, re, shutil, threading, time
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import devices
from chat_engine import ROOT, RUNS, Engine, gguf_path, list_models, weights_stamp
from data import crawl
from studio import auth, db, feedback, gate, jobs
from writing import formats as F
from studio import styles
from studio import copilot
from core import backends as core_backends, commands as core_commands, net as core_net, parallel as core_parallel, paths as core_paths

PY = jobs.PY
WEB = os.path.join(ROOT, "web")
app = FastAPI(title="Text LLM Studio")
app.include_router(copilot.router)  # Training → Copilot: code sources, folder picker, code tokenizer


def err(msg, code=400):
    raise HTTPException(code, msg)


def _launch(*a, **kw):
    try:
        return jobs.launch(*a, **kw)
    except RuntimeError as e:
        err(str(e), 409)


def run_name(raw, default="my-model"):
    """A run name from the browser: letters, digits, - _ . only, and never one of the Studio's own
    _folders (runs/_uploads, runs/_cluster, ...)."""
    name = core_paths.safe_name(raw, default)
    return default if name.startswith("_") else name


def run_dir(name):
    """The run folder for an existing model named by the browser; 404 for anything else."""
    d = os.path.join(RUNS, run_name(name or "", default=""))
    if not name or run_name(name, default="") != name or not os.path.isdir(d):
        err(f"no run named {name}", 404)
    return d


# ---------------------------------------------------------------- accounts & roles
# Every /api route needs a signed-in user. Admins can call all of them; regular users only
# the ones below (Chat and their own Settings). New routes are admin-only unless listed here.
PUBLIC_ROUTES = {("GET", "/api/auth/state"), ("POST", "/api/auth/login"), ("POST", "/api/auth/setup"),
                 ("POST", "/api/auth/logout")}
USER_ROUTES = [(m, re.compile(p)) for m, p in [
    ("GET", r"/api/me"), ("POST", r"/api/me"), ("GET", r"/api/status"), ("GET", r"/api/models"),
    ("GET", r"/api/conversations"), ("GET", r"/api/conversations/\d+"), ("DELETE", r"/api/conversations/\d+"),
    ("POST", r"/api/chat"), ("POST", r"/api/messages/\d+/rate"), ("GET", r"/api/formats"),
    ("POST", r"/api/code/complete"),
]]


def _same_origin(request):
    """A browser sends Origin on cross-site requests; a different host means another site is trying
    to use this person's cookie (CSRF). Same-site requests and non-browser clients pass."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin == "null":  # a sandboxed frame or a file:// page: never one of our own
        return False
    # behind a reverse proxy the browser's host arrives in X-Forwarded-Host
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    return urlsplit(origin).netloc.lower() == host.split(",")[0].strip().lower()


@app.middleware("http")
async def require_login(request: Request, call_next):
    path, method = request.url.path, request.method
    if not path.startswith("/api/") or (method, path) in PUBLIC_ROUTES:
        resp = await call_next(request)
    else:
        resp = await _api_request(request, call_next, path, method)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    return resp


async def _api_request(request, call_next, path, method):
    if method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
        return JSONResponse({"detail": "Cross-site requests are not allowed."}, status_code=403)
    user = auth.user_for_token(request.cookies.get(auth.COOKIE))
    if not user:
        return JSONResponse({"detail": "Please sign in."}, status_code=401)
    if user["role"] != "admin" and not any(m == method and p.fullmatch(path) for m, p in USER_ROUTES):
        return JSONResponse({"detail": "Only admins can do that."}, status_code=403)
    request.state.user = user
    return await call_next(request)


def me(request: Request):
    return request.state.user


def is_admin(request):
    return me(request)["role"] == "admin"


def own_conversation(request, cid):
    """The conversation if the signed-in user may see it (their own, or any for admins)."""
    c = db.q("SELECT * FROM conversations WHERE id=?", (cid,), one=True)
    if not c or (c["user_id"] != me(request)["id"] and not is_admin(request)):
        err("not found", 404)
    return c


def _set_cookie(resp, token, request=None):
    https = bool(request) and (request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    resp.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400, httponly=True, samesite="lax",
                    secure=https)


class LoginReq(BaseModel):
    username: str
    password: str
    display_name: str | None = None


@app.get("/api/auth/state")
def auth_state(request: Request):
    return {"needs_setup": auth.count_users() == 0,
            "user": auth.user_for_token(request.cookies.get(auth.COOKIE))}


@app.post("/api/auth/setup")
def auth_setup(r: LoginReq, request: Request, response: Response):
    """First run only: create the admin account."""
    if auth.count_users():  # quick answer; create_user re-checks atomically
        err("Accounts already exist. Sign in instead.", 409)
    try:
        auth.create_user(r.username, r.password, "admin", r.display_name, only_if_first=True)
    except auth.NotFirst:
        err("Accounts already exist. Sign in instead.", 409)
    except ValueError as e:
        err(str(e))
    token, user = auth.login(r.username, r.password)
    _set_cookie(response, token, request)
    return user


@app.post("/api/auth/login")
def auth_login(r: LoginReq, request: Request, response: Response):
    client = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (request.client.host if request.client else "")
    if auth.locked_out(r.username, client):
        err("Too many wrong passwords. Wait a minute and try again.", 429)
    token, user = auth.login(r.username, r.password, client=client)
    if not token:
        err("Wrong username or password.", 401)
    _set_cookie(response, token, request)
    return user


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    auth.logout(request.cookies.get(auth.COOKIE))
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/api/me")
def get_me(request: Request):
    return me(request)


class MeReq(BaseModel):
    display_name: str | None = None
    prefs: dict | None = None
    current_password: str | None = None
    new_password: str | None = None


@app.post("/api/me")
def update_me(r: MeReq, request: Request, response: Response):
    u = me(request)
    if r.prefs is not None:
        p = r.prefs
        try:
            if "temperature" in p:
                p["temperature"] = min(1.5, max(0.0, float(p["temperature"])))
            if "max_tokens" in p:
                p["max_tokens"] = min(1024, max(16, int(p["max_tokens"])))
            if "model" in p:
                p["model"] = str(p["model"])[:200]
        except (TypeError, ValueError):
            err("Invalid settings.")
    if r.new_password is not None:
        if not auth.check_password(r.current_password or "", auth.get_user(u["id"])["pw_hash"]):
            err("Your current password isn't right.")
    try:
        out = auth.update_user(u["id"], display_name=r.display_name, prefs=r.prefs, password=r.new_password)
    except ValueError as e:
        err(str(e))
    if r.new_password is not None:  # changing the password signs out other sessions; keep this one
        token, _ = auth.login(u["username"], r.new_password)
        _set_cookie(response, token, request)
    return out


@app.get("/api/users")
def users():
    return auth.list_users()


class UserReq(BaseModel):
    username: str = ""
    password: str | None = None
    role: str | None = None
    display_name: str | None = None


@app.post("/api/users")
def add_user(r: UserReq):
    try:
        return auth.create_user(r.username, r.password or "", r.role or "user", r.display_name)
    except ValueError as e:
        err(str(e))


@app.patch("/api/users/{uid}")
def edit_user(uid: int, r: UserReq, request: Request):
    if uid == me(request)["id"] and r.role and r.role != "admin":
        err("You can't remove your own admin role.")
    try:
        return auth.update_user(uid, role=r.role, password=r.password or None, display_name=r.display_name)
    except LookupError as e:
        err(str(e), 404)
    except ValueError as e:
        err(str(e))


@app.delete("/api/users/{uid}")
def remove_user(uid: int, request: Request):
    if uid == me(request)["id"]:
        err("You can't delete your own account.")
    try:
        auth.delete_user(uid)
    except LookupError as e:
        err(str(e), 404)
    except ValueError as e:
        err(str(e))
    return {"ok": True}


# ---------------------------------------------------------------- status / data
_DEVICE = None


def device_info():
    global _DEVICE
    if _DEVICE is None:
        # doesn't change while the app runs, except when the device setting changes (apply_device)
        _DEVICE = {**devices.info(), "choice": db.get_settings().get("device", "auto"),
                   "choices": devices.available(), "engines": core_backends.report()}
    return _DEVICE


_ENV_DEVICE = os.environ.get("TEXTLLM_DEVICE")  # set outside the Studio: wins while the setting is "auto"
_ENV_PRIVATE_URLS = os.environ.get("TEXTLLM_ALLOW_PRIVATE_URLS")  # same for local-network links


def apply_private_urls(allow):
    """Links into the local network (core/net.py reads TEXTLLM_ALLOW_PRIVATE_URLS; jobs inherit it).
    Like the device: the environment set before starting the Studio wins while the setting is off."""
    if allow or _ENV_PRIVATE_URLS:
        os.environ["TEXTLLM_ALLOW_PRIVATE_URLS"] = "1" if allow else _ENV_PRIVATE_URLS
    else:
        os.environ.pop("TEXTLLM_ALLOW_PRIVATE_URLS", None)


def apply_device(dev):
    """Send training jobs (they inherit os.environ) and chat to this device; see devices.pick()."""
    global _DEVICE
    if dev and dev != "auto":
        os.environ["TEXTLLM_DEVICE"] = dev
    elif _ENV_DEVICE:
        os.environ["TEXTLLM_DEVICE"] = _ENV_DEVICE
    else:
        os.environ.pop("TEXTLLM_DEVICE", None)
    _DEVICE = None
    with _engine_lock:
        ENGINES.clear()  # chat models reload on the new device


@app.get("/api/status")
def status(request: Request):
    s = db.get_settings()
    if s["auto_train"] and not jobs.running():  # ratings that arrived while another job ran
        for m in {r["model"] for r in db.q("SELECT DISTINCT model FROM messages WHERE role='assistant' "
                                           "AND trained_in IS NULL AND (rating IS NOT NULL OR correction IS NOT NULL)")}:
            failed = [j for j in jobs.history(20) if j["kind"] == "feedback" and j["run"] == m
                      and j["exit"] not in (0, None) and time.time() - j["ended"] < 900]
            if m and not failed and os.path.isdir(os.path.join(RUNS, m)) and maybe_auto_train(m):
                break
    if not is_admin(request):  # users don't see jobs, data or the feedback queue
        return {"user": me(request)}
    return {"job": jobs.status(), "data": jobs.data_status(), "settings": s, "device": device_info(),
            "pending_feedback": db.pending_feedback(), "user": me(request)}


class DataReq(BaseModel):
    step: str            # download | tokenizer | tokcheck | encode | instructions
    books: int = 3000
    web_gb: float = 0.3
    # more rights-cleared sources, 0 = skip (see data/build_corpus.py for licenses)
    gutenberg_method: str = "mirror"   # mirror | harvest
    pg19_books: int = 0
    standard_ebooks: int = 0
    common_corpus_gb: float = 0
    archive_books: int = 0
    cosmopedia_gb: float = 0
    cosmopedia_subsets: str = "stories"
    wikipedia_gb: float = 0
    wikipedia_lang: str = "en"
    common_pile_gb: float = 0
    common_pile_subsets: str = "news,public_domain_review,pressbooks,cccc"
    vocab: int = 8192
    hf_sets: bool = True
    # downloading (flaky connections: "download" keeps partial files and resumes)
    fineweb_mode: str = "stream"       # stream | download
    keep_parquet: bool = False
    retries: int = 30
    workers: int = 8
    common_corpus_lang: str = "en"
    # tokenizer
    sample_files: int = 40
    tok_style: str = "modern"          # modern (v2) | gpt2 (v1, as before)
    lang_balance: float = 0.3          # 1 = the corpus as it is
    check_mb: float = 4.0              # MB of text the quality check reads
    val_frac: float = 0.005
    # writing instructions
    inst_gutenberg: bool = True
    inst_stories: int = 1200
    inst_essays: int = 1200
    inst_books: int = 300
    inst_my_posts: bool = True         # the files uploaded under "Your own posts"
    synth_model: str = ""              # Apache 2.0 model (HF id or .gguf path) that writes example posts
    synth_per_format: int = 200
    backtranslate: str = ""            # model that writes the request for each text
    seed: int = 0


MY_POSTS = os.path.join(ROOT, "data", "my_posts")


def model_ref(v, what):
    """A Hugging Face id or a local path, never something that looks like a command-line flag."""
    v = (v or "").strip().strip('"')
    if v and (v.startswith("-") or not re.fullmatch(r"[\w .:/\\~@+-]{1,300}", v)):
        err(f"{what}: give a Hugging Face model id or a file path")
    return v


def my_posts_files():
    return sorted(glob.glob(os.path.join(MY_POSTS, "*.txt")) + glob.glob(os.path.join(MY_POSTS, "*.jsonl")))


@app.post("/api/data")
def data(r: DataReq):
    inst = [PY, "data/build_instructions.py", "--from-studio", "--stories", str(max(0, r.inst_stories)),
            "--essays", str(max(0, r.inst_essays)), "--books", str(max(1, r.inst_books)), "--seed", str(r.seed)]
    inst += [] if r.inst_gutenberg else ["--no-gutenberg"]
    inst += ["--hf-sets"] if r.hf_sets else []
    if r.inst_my_posts:
        for p in my_posts_files():
            inst += ["--my-posts", p]
    synth, bt = model_ref(r.synth_model, "Synthetic posts model"), model_ref(r.backtranslate, "Back-translation model")
    if synth:
        inst += ["--synth-model", synth, "--synth-per-format", str(max(1, min(5000, r.synth_per_format)))]
    if bt:
        inst += ["--backtranslate", bt]
    cmds = {
        "download": [PY, "data/build_corpus.py", "--gutenberg-max", str(r.books), "--fineweb-gb", str(r.web_gb),
                     "--gutenberg-method", r.gutenberg_method if r.gutenberg_method in ("mirror", "harvest") else "mirror",
                     "--pg19-books", str(max(0, r.pg19_books)), "--standard-ebooks", str(max(0, r.standard_ebooks)),
                     "--common-corpus-gb", str(max(0.0, r.common_corpus_gb)),
                     "--archive-books", str(max(0, r.archive_books)),
                     "--cosmopedia-gb", str(max(0.0, r.cosmopedia_gb)),
                     "--cosmopedia-subsets", re.sub(r"[^a-z0-9_,]", "", r.cosmopedia_subsets) or "stories",
                     "--wikipedia-gb", str(max(0.0, r.wikipedia_gb)),
                     "--wikipedia-lang", re.sub(r"[^a-z0-9-]", "", r.wikipedia_lang.lower()) or "en",
                     "--common-pile-gb", str(max(0.0, r.common_pile_gb)),
                     "--common-pile-subsets", re.sub(r"[^a-z0-9_,]", "", r.common_pile_subsets)
                     or "news,public_domain_review,pressbooks,cccc",
                     "--common-corpus-lang", re.sub(r"[^a-z-]", "", r.common_corpus_lang.lower()) or "en",
                     "--fineweb-mode", "download" if r.fineweb_mode == "download" else "stream",
                     "--retries", str(max(0, min(1000, r.retries))), "--workers", str(max(1, min(64, r.workers)))]
                    + (["--keep-parquet"] if r.keep_parquet else []),
        "tokenizer": [PY, "tokenizer/train_tokenizer.py", "--vocab", str(max(1000, min(65000, r.vocab))),
                      "--sample-files", str(max(1, r.sample_files)),
                      "--style", r.tok_style if r.tok_style in ("modern", "gpt2") else "modern",
                      "--lang-balance", str(max(0.0, min(1.0, r.lang_balance))),
                      "--sample-mb", str(max(0.0, min(1024.0, r.check_mb)))],
        # quality numbers for the current tokenizer (and the one it replaced): Data, step 2
        "tokcheck": [PY, "tokenizer/train_tokenizer.py", "--check", "--sample-mb", str(max(0.5, min(1024.0, r.check_mb)))],
        "encode": [PY, "tokenizer/train_tokenizer.py", "--encode", "--val-frac", str(max(0.0001, min(0.5, r.val_frac)))],
        # writing instructions for fine-tuning: posts, stories, essays (rights-cleared sources only)
        "instructions": inst,
    }
    if r.step not in cmds:
        err("unknown step")
    return _launch("data", r.step, cmds[r.step], os.path.join(RUNS, "_data", "log.txt"))


@app.get("/api/data/my-posts")
def my_posts_list():
    return [{"name": os.path.basename(p), "kb": round(os.path.getsize(p) / 1024, 1)} for p in my_posts_files()]


@app.post("/api/data/my-posts")
async def my_posts_add(files: list[UploadFile] = File(default=[])):
    """Your own posts for the writing instructions: .txt (posts separated by --- lines) or .jsonl."""
    os.makedirs(MY_POSTS, exist_ok=True)
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith((".txt", ".jsonl")):
            err(f"{name or 'file'}: use a .txt file (posts separated by --- lines) or .jsonl")
        with open(os.path.join(MY_POSTS, core_paths.safe_name(name, "posts.txt")), "wb") as o:
            shutil.copyfileobj(f.file, o)
    return my_posts_list()


@app.delete("/api/data/my-posts/{name}")
def my_posts_delete(name: str):
    p = os.path.join(MY_POSTS, os.path.basename(name))
    if not os.path.isfile(p):
        err("no such file", 404)
    os.remove(p)
    return my_posts_list()


@app.post("/api/stop")
def stop():
    return jobs.stop()


# ---------------------------------------------------------------- training
class TrainReq(BaseModel):
    mode: str = "scratch"          # scratch | lora
    name: str = "my-model"
    resume: bool = False
    preset: str = "cpu"
    vocab: int = 8192
    steps: int = 20000
    batch: int = 16
    accum: int = 4
    lr: float = 1e-3
    base: str = "Qwen/Qwen3-0.6B"
    examples: int = 4000
    seq: int = 512
    instructions: bool = True     # use data/instructions/train.jsonl if it exists
    book_share: float = 0.2       # share of book-continuation examples in the mix
    lora_r: int = 8
    epochs: float = 0             # > 0: train this many passes instead of a fixed number of steps
    gguf: bool = True             # make a fast quantized CPU copy after training
    # from scratch, advanced (0 = the script's default / automatic)
    warmup: int = 0
    min_lr: float = 0
    weight_decay: float = 0.1
    compile: bool = False
    procs: int = 0                # training processes on this PC (0 = auto: the speed test's pick, else 1)
    threads: int = 0              # CPU threads for the run (0 = all physical cores); also for LoRA
    gpu_wait: float = 20          # minutes the first GPU step may take before the run moves to the CPU (0 = wait)
    # LoRA, advanced
    init_from: str = ""           # continue from this fine-tuned run's adapter (same base model)
    pairs_per_doc: int = 8
    min_doc_chars: int = 5000
    # copilot (code) models: Training → Copilot
    task: str = "writing"         # writing | code
    code_examples: int = 4000
    fim_rate: float = 0.5         # share of fill-in-the-middle examples
    code_chat_share: float = 0.2  # share of chat-form examples (Chat → Code assistant)
    code_langs: str = ""          # only these languages (empty = all collected)


@app.post("/api/train")
def train(r: TrainReq):
    name = run_name(r.name)
    out = os.path.join(RUNS, name)
    code = r.task == "code"
    resuming_saved = r.resume and os.path.exists(os.path.join(out, "train_cmd.json"))
    if r.resume and not resuming_saved and ((_read_json(os.path.join(out, "meta.json")) or {}).get("retrained") or {}).get("resumable") is False:
        # a retrain on your own files and links keeps no copy of them, so it must not quietly go on with other data
        err(f"{name} was retrained on files and links that were deleted afterwards, so it can't be resumed. "
            "Start the retrain again (Training → Retrain).", 409)
    if r.mode == "scratch":
        bin_dir, tok_src = ("data/bin-code", "tokenizer/bpe-code.json") if code else ("data/bin", "tokenizer/bpe.json")
        where = "Training → Copilot, steps 1–3" if code else "Training → Data, steps 1–3"
        if not resuming_saved and not os.path.exists(os.path.join(ROOT, bin_dir, "train.bin")):
            err(f"Prepare the training data first ({where}).")
        os.makedirs(out, exist_ok=True)
        tok_copy = os.path.join(out, "tokenizer.json")
        if not (r.resume and os.path.exists(tok_copy)):
            # Keep the tokenizer with the weights, so the run folder can be moved to another PC.
            src = os.path.join(ROOT, tok_src)
            if not os.path.exists(src):
                err(f"Train the tokenizer first ({where.replace('steps 1–3', 'step 2')}).")
            shutil.copyfile(src, tok_copy)
        info = jobs.tokenizer_info(tok_copy)
        if info and info.get("vocab"):
            r.vocab = info["vocab"]           # the model must match its tokenizer (train.py keeps a resumed run's size)
        meta_p = os.path.join(out, "meta.json")
        if not (r.resume and os.path.exists(meta_p)):
            json.dump({"type": "scratch", "tok": "tokenizer.json", "preset": r.preset,
                       "tokenizer": jobs.tokenizer_info(tok_copy), **({"task": "code"} if code else {})},
                      open(meta_p, "w"))
        cmd = [PY, "train.py", "--out", out, "--data", bin_dir, "--preset", r.preset, "--vocab", str(r.vocab),
               "--max-steps", str(r.steps), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--warmup", str(r.warmup if r.warmup > 0 else max(10, r.steps // 40)),
               "--eval-every", str(max(10, r.steps // 80)), "--weight-decay", str(max(0.0, r.weight_decay))]
        if r.min_lr > 0:
            cmd += ["--min-lr", str(min(r.min_lr, r.lr))]
        if r.compile:
            cmd.append("--compile")
        cmd += ["--procs", str(max(0, min(64, r.procs))), "--gpu-wait", str(max(0.0, min(1440.0, r.gpu_wait)))]
        if r.threads > 0:
            cmd += ["--threads", str(min(1024, r.threads))]
        if r.resume:
            cmd.append("--resume")
    else:
        import glob
        inst = os.path.join(ROOT, "data/instructions/train.jsonl")
        use_inst = r.instructions and os.path.exists(inst)
        if code:
            use_inst = False
            if not resuming_saved and not copilot.has_code():
                err("Collect some code first (Training → Copilot, step 1).")
        elif not resuming_saved and not use_inst and not glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True):
            err("Add some text first: build writing instructions or download books (Training → Data).")
        os.makedirs(out, exist_ok=True)
        cmd = [PY, "finetune/finetune_lora.py", "--out", out, "--base", r.base, "--examples", str(r.examples),
               "--seq", str(r.seq), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--max-steps", "0" if r.epochs > 0 else str(r.steps), "--eval-every", str(max(5, r.steps // 20)),
               "--save-every", str(max(5, r.steps // 10)), "--lora-r", str(r.lora_r),
               "--pairs-per-doc", str(max(1, r.pairs_per_doc)), "--min-doc-chars", str(max(0, r.min_doc_chars))]
        if r.init_from and not r.resume:
            src = run_dir(r.init_from)
            ad = os.path.join(src, "adapter")
            if not os.path.isdir(ad) or src == out:
                err(f"{r.init_from} has no LoRA adapter to start from.")
            src_base = json.load(open(os.path.join(src, "meta.json"))).get("base")
            if src_base and src_base != r.base:
                err(f"{r.init_from} was trained on {src_base}; pick that base model to continue from it.")
            cmd += ["--init-adapter", ad]
        if r.epochs > 0:
            cmd += ["--epochs", str(r.epochs)]
        if use_inst:
            cmd += ["--instructions", inst, "--book-share", str(r.book_share)]
        if code:
            cmd += ["--code", "data/code", "--code-examples", str(max(1, r.code_examples)),
                    "--fim-rate", str(max(0.0, min(1.0, r.fim_rate))),
                    "--code-chat-share", str(max(0.0, min(1.0, r.code_chat_share)))]
            langs = re.sub(r"[^\w#+, -]", "", r.code_langs).strip()
            if langs:
                cmd += ["--code-langs", langs]
        if r.gguf:
            cmd += ["--gguf"]
        if r.threads > 0:
            cmd += ["--threads", str(min(1024, r.threads))]
        if r.resume:
            cmd.append("--resume")
    saved_cmd = os.path.join(out, "train_cmd.json")
    if r.resume and os.path.exists(saved_cmd):
        # Continue with the run's own settings (LR schedule, batch, data), not today's form values.
        try:
            cmd = [PY] + core_commands.check(json.load(open(saved_cmd)))
        except (ValueError, OSError) as e:  # a bundle from elsewhere with an odd command: don't run it
            err(f"This run's saved training command can't be used: {e}")
        cmd = [out if i > 0 and cmd[i - 1] == "--out" else c for i, c in enumerate(cmd)]
        if "--resume" not in cmd:
            cmd.append("--resume")
        # Threads and processes belong to this PC, not the run (it may have come from another one).
        speed = {"--procs": str(max(0, min(64, r.procs))) if cmd[1] == "train.py" else None,
                 "--threads": str(min(1024, r.threads)) if r.threads > 0 else None}
        for flag, val in speed.items():
            while flag in cmd:
                i = cmd.index(flag)
                del cmd[i:i + 2]
            if val is not None:
                cmd += [flag, val]
        if "--max-steps" in cmd:  # "train longer": a bigger step count in the form extends the run
            i = cmd.index("--max-steps") + 1
            if int(cmd[i]) > 0 and r.steps > int(cmd[i]):
                cmd[i] = str(r.steps)
    else:
        rel = lambda c: os.path.relpath(c, ROOT).replace(os.sep, "/") if os.path.isabs(c) and c.startswith(ROOT) else c
        json.dump([rel(c) for c in cmd[1:] if c != "--resume"], open(saved_cmd, "w"))
    mp = os.path.join(out, "metrics.jsonl")
    if not r.resume and os.path.exists(mp):
        os.rename(mp, os.path.join(out, f"metrics.{int(time.time())}.jsonl"))
    return _launch("training", name, cmd, os.path.join(out, "log.txt"), run=name)


# ---------------------------------------------------------------- several processes on one PC
class SpeedReq(BaseModel):
    preset: str = "cpu"
    vocab: int = 8192
    batch: int = 16
    threads: int = 0


@app.get("/api/train/parallel")
def train_parallel():
    """The speed test results (fastest process count per model size) and this PC's cores."""
    return {"bench": core_parallel.load_bench(), "cores": core_parallel.physical_cores(),
            "logical": os.cpu_count() or 1}


@app.post("/api/train/speedtest")
def train_speedtest(r: SpeedReq):
    """Train a few steps with 1, 2 and 4 processes on random tokens; Auto then uses the fastest."""
    from model.gpt import PRESETS

    if r.preset not in PRESETS:
        err("unknown model size")
    cmd = [PY, "-m", "core.parallel", "--preset", r.preset, "--vocab", str(max(256, min(65535, r.vocab))),
           "--batch", str(max(1, min(512, r.batch))), "--steps", "30"]
    if r.threads > 0:
        cmd += ["--threads", str(min(1024, r.threads))]
    return _launch("speedtest", f"speed test ({r.preset})", cmd, os.path.join(RUNS, "_parallel.log"))


# ---------------------------------------------------------------- cluster (several PCs train one model)
CLUSTER_DIR = os.path.join(RUNS, "_cluster")
CLUSTER_LAST = os.path.join(CLUSTER_DIR, "last.json")       # the coordinator this PC started last
WORKER_STATUS = os.path.join(CLUSTER_DIR, "worker.json")    # written by cluster.py worker


class ClusterReq(BaseModel):
    name: str = "cluster-writer"
    resume: bool = False
    preset: str = "cpu"
    vocab: int = 8192
    rounds: int = 200
    round_minutes: float = 5
    lr: float = 1e-3
    min_lr: float = 0
    warmup: int = 200
    weight_decay: float = 0.1
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True
    min_workers: int = 1
    eval_every: int = 1
    port: int = 7861
    secret: str = ""
    task: str = "writing"         # writing | code (the copilot's data/bin-code and code tokenizer)
    local_worker: bool = True     # this PC trains too
    upnp: bool = True             # ask the router to open the port for PCs on other networks
    batch: int = 16               # this PC's worker
    accum: int = 4


class JoinReq(BaseModel):
    address: str
    secret: str
    name: str = ""
    batch: int = 0                # 0 = 16 on a GPU, 8 on a CPU
    accum: int = 4
    data_mb: float = 200
    local_data: bool = False      # use this PC's data/bin instead of downloading a share
    task: str = "writing"         # code: local_data means data/bin-code
    reserve: float = 0


def _read_json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


@app.post("/api/cluster/start")
def cluster_start(r: ClusterReq):
    import secrets as _secrets

    name = run_name(r.name, "cluster-writer")
    out = os.path.join(RUNS, name)
    resuming = r.resume and os.path.exists(os.path.join(out, "ckpt.pt"))
    code = r.task == "code"
    bin_dir, tok_src = ("data/bin-code", "tokenizer/bpe-code.json") if code else ("data/bin", "tokenizer/bpe.json")
    where = "Training → Copilot" if code else "Training → Data"
    if not os.path.exists(os.path.join(ROOT, bin_dir, "train.bin")):
        err(f"Prepare the training data first ({where}, steps 1–3). Workers get their share from this PC.")
    os.makedirs(out, exist_ok=True)
    tok_copy = os.path.join(out, "tokenizer.json")
    if not (resuming and os.path.exists(tok_copy)):
        src = os.path.join(ROOT, tok_src)
        if not os.path.exists(src):
            err(f"Train the tokenizer first ({where}, step 2).")
        shutil.copyfile(src, tok_copy)
    if code:
        from tokenizers import Tokenizer

        r.vocab = Tokenizer.from_file(tok_copy).get_vocab_size()
    meta_p = os.path.join(out, "meta.json")
    if not (resuming and os.path.exists(meta_p)):
        json.dump({"type": "scratch", "tok": "tokenizer.json", "preset": r.preset, "cluster": True,
                   "tokenizer": jobs.tokenizer_info(tok_copy),
                   **({"task": "code"} if code else {})}, open(meta_p, "w"))
    secrets_p = os.path.join(CLUSTER_DIR, "secrets.json")   # run name -> secret, so a resume keeps it
    known = _read_json(secrets_p) or {}
    # Resuming keeps the run's secret (workers still hold it and reconnect by themselves).
    secret = r.secret.strip() or (known.get(name) if resuming else None) or "-".join(_secrets.token_hex(4) for _ in range(4))
    cmd = [PY, "cluster.py", "coordinator", "--out", out, "--data", bin_dir, "--preset", r.preset, "--vocab", str(r.vocab),
           "--rounds", str(max(1, r.rounds)), "--round-minutes", str(max(0.2, r.round_minutes)), "--lr", str(r.lr),
           "--min-lr", str(min(r.min_lr, r.lr) if r.min_lr > 0 else r.lr / 10), "--warmup", str(max(0, r.warmup)),
           "--weight-decay", str(max(0.0, r.weight_decay)), "--outer-lr", str(r.outer_lr),
           "--outer-momentum", str(r.outer_momentum), "--min-workers", str(max(1, r.min_workers)),
           "--eval-every", str(max(1, r.eval_every)), "--port", str(r.port),
           "--batch", str(max(1, r.batch)), "--grad-accum", str(max(1, r.accum)),
           "--local-device", db.get_settings().get("device", "auto") or "auto"]
    if not r.nesterov:
        cmd.append("--no-nesterov")
    if not r.upnp:
        cmd.append("--no-upnp")
    if r.local_worker:
        cmd.append("--local-worker")
    if resuming:
        cmd.append("--resume")
    mp = os.path.join(out, "metrics.jsonl")
    if not resuming and os.path.exists(mp):
        os.rename(mp, os.path.join(out, f"metrics.{int(time.time())}.jsonl"))
    for f in ("cluster.json",):
        try:
            os.remove(os.path.join(out, f))
        except OSError:
            pass
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    json.dump({"run": name, "secret": secret, "port": r.port, "started": time.time()}, open(CLUSTER_LAST, "w"))
    json.dump({**known, name: secret}, open(secrets_p, "w"))
    # the secret goes through the environment, so it isn't written into log.txt with the command
    return _launch("cluster", name, cmd, os.path.join(out, "log.txt"), run=name, env={"TEXTLLM_CLUSTER_SECRET": secret})


@app.post("/api/cluster/join")
def cluster_join(r: JoinReq):
    addr = r.address.strip()
    if not addr or not r.secret.strip():
        err("Enter the coordinator's address and its secret.")
    os.makedirs(CLUSTER_DIR, exist_ok=True)
    try:
        os.remove(WORKER_STATUS)
    except OSError:
        pass
    cmd = [PY, "cluster.py", "worker", "--join", addr,
           "--device", db.get_settings().get("device", "auto") or "auto", "--batch", str(max(0, r.batch)),
           "--grad-accum", str(max(1, r.accum)), "--data-mb", str(max(1.0, r.data_mb)),
           "--reserve", str(max(0.0, r.reserve)), "--status-file", WORKER_STATUS]
    if r.name.strip():
        cmd += ["--name", r.name.strip()[:40]]
    if r.local_data:
        bin_dir = "data/bin-code" if r.task == "code" else "data/bin"
        if not os.path.exists(os.path.join(ROOT, bin_dir, "train.bin")):
            err(f"This PC has no prepared data ({bin_dir}/train.bin). Untick 'use this PC's data' to download a share.")
        cmd += ["--data", os.path.join(ROOT, bin_dir)]
    json.dump({"address": addr, "name": r.name}, open(os.path.join(CLUSTER_DIR, "joined.json"), "w"))
    return _launch("worker", addr, cmd, os.path.join(CLUSTER_DIR, "worker-log.txt"),
                   env={"TEXTLLM_CLUSTER_SECRET": r.secret.strip()})


@app.get("/api/cluster")
def cluster_state():
    last = _read_json(CLUSTER_LAST)
    coord = None
    if last:
        d = os.path.join(RUNS, os.path.basename(last["run"]))
        coord = {**(_read_json(os.path.join(d, "cluster.json")) or {}), "run": last["run"], "secret": last["secret"],
                 "port": last["port"], "running": jobs.running() and jobs.JOB["kind"] == "cluster",
                 "log": jobs.tail(os.path.join(d, "log.txt"), 80)}
    worker = {**(_read_json(WORKER_STATUS) or {}), "running": jobs.running() and jobs.JOB["kind"] == "worker",
              "last": _read_json(os.path.join(CLUSTER_DIR, "joined.json")),
              "log": jobs.tail(os.path.join(CLUSTER_DIR, "worker-log.txt"), 80)}
    return {"coordinator": coord, "worker": worker, "data": jobs.data_status()}


@app.get("/api/cluster/phone")
def cluster_phone(addr: str):
    """The join page for phones, as a QR code (SVG) plus the Termux command."""
    import cluster as cl

    last = _read_json(CLUSTER_LAST)
    if not last:
        err("Start a cluster first.")
    if not re.fullmatch(r"https?://[\w.\-\[\]:]+", addr):
        err("bad address")
    from urllib.parse import quote

    url = f"{addr}/phone?secret={quote(last['secret'], safe='')}"
    svg = None
    try:
        import io as _io

        import segno

        buf = _io.BytesIO()
        segno.make(url, error="m").save(buf, kind="svg", scale=5, border=2, dark="#111", light="#fff")
        svg = buf.getvalue().decode()
    except ImportError:
        pass
    return {"url": url, "svg": svg, "command": cl.phone_command(addr, last["secret"])}


# ---------------------------------------------------------------- style profiles (write like an account)
@app.get("/api/styles")
def styles_list():
    from data import own_posts

    out = []
    for p in own_posts.list_profiles():
        rd = os.path.join(RUNS, styles.run_name(p["id"]))
        out.append({**p, "run": styles.run_name(p["id"]), "trained": os.path.isdir(os.path.join(rd, "adapter")),
                    "training": jobs.running() and jobs.JOB["run"] == styles.run_name(p["id"])})
    return {"profiles": out, "kinds": own_posts.KINDS, "not_possible": own_posts.NOT_POSSIBLE,
            "min_posts": styles.MIN_POSTS}


@app.post("/api/styles/import")
async def styles_import(name: str = Form(...), kind: str = Form(...), mine: bool = Form(False),
                        url: str = Form(""), page_id: str = Form(""), token: str = Form(""),
                        keep_replies: bool = Form(False), profile: str = Form(""),
                        file: UploadFile | None = File(default=None)):
    """Import posts into a style profile. An uploaded export is read, then deleted right away."""
    from data import own_posts

    pid = own_posts.slug(profile or name)
    path = None
    if kind in ("x-archive", "facebook-export", "instagram-export", "mastodon-archive"):
        if not file or not file.filename:
            err("Choose the export .zip file.")
        up = os.path.join(RUNS, "_uploads")
        os.makedirs(up, exist_ok=True)
        path = os.path.join(up, f"style-{int(time.time() * 1000)}.zip")
        with open(path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
    try:
        import asyncio

        return await asyncio.to_thread(own_posts.import_source, pid, name, kind, mine, path, url.strip() or None,
                                       page_id or None, token or None, keep_replies)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        err(str(e))
    finally:  # the export holds private data (messages, contacts): never keep it
        if path and os.path.exists(path):
            os.remove(path)


@app.get("/api/styles/{pid}/sample")
def styles_sample(pid: str, n: int = 5):
    from data import own_posts

    import random

    posts = own_posts.load_posts(own_posts.slug(pid))
    return random.sample(posts, min(n, len(posts)))


@app.delete("/api/styles/{pid}")
def styles_delete(pid: str, model: bool = False):
    from data import own_posts

    pid = own_posts.slug(pid)
    if jobs.running() and jobs.JOB["run"] == styles.run_name(pid):
        err("Stop the training first.", 409)
    try:
        own_posts.delete_profile(pid)
    except LookupError as e:
        err(str(e), 404)
    if model:
        shutil.rmtree(os.path.join(RUNS, styles.run_name(pid)), ignore_errors=True)
    return {"ok": True}


class StyleTrainReq(BaseModel):
    start_from: str = ""      # a fine-tuned run, or "" for the base model
    epochs: float = 3.0
    gguf: bool = True


@app.post("/api/styles/{pid}/train")
def styles_train(pid: str, r: StyleTrainReq):
    from data import own_posts

    pid = own_posts.slug(pid)
    try:
        cmd, out = styles.train_command(pid, PY, RUNS, run_dir(r.start_from) if r.start_from else None,
                                        max(0.5, min(10, r.epochs)), r.gguf)
    except LookupError as e:
        err(str(e), 404)
    except ValueError as e:
        err(str(e))
    # saved like /api/train does, so Resume on the Runs page continues with these settings (and --meta-extra)
    rel = lambda c: os.path.relpath(c, ROOT).replace(os.sep, "/") if os.path.isabs(c) and c.startswith(ROOT) else c
    json.dump([rel(c) for c in cmd[1:]], open(os.path.join(out, "train_cmd.json"), "w"))
    mp = os.path.join(out, "metrics.jsonl")
    if os.path.exists(mp):  # a fresh style run: keep the old curve aside, as /api/train does
        os.rename(mp, os.path.join(out, f"metrics.{int(time.time())}.jsonl"))
    return _launch("training", styles.run_name(pid), cmd, os.path.join(out, "log.txt"), run=styles.run_name(pid))


@app.post("/api/texts")
async def texts(files: list[UploadFile] = File(default=[]), urls: str = Form(""), paths: str = Form(""),
                action: str = Form("learn"), model: str = Form(""), passes: float = Form(3),
                delete_local: bool = Form(False), mix: float = Form(1.0), lr: float = Form(0),
                ocr: str = Form("auto"), ocr_lang: str = Form("auto")):
    from data import pdftext

    up_dir = os.path.join(RUNS, "_uploads", str(int(time.time() * 1000)))
    saved, skipped = [], []
    for f in files:
        if not f.filename:
            continue
        os.makedirs(up_dir, exist_ok=True)
        # keep letters of any script (Armenian, Cyrillic, ...) but never "..", a drive or a folder
        base = os.path.basename(f.filename.replace("\\", "/")).strip().lstrip(".") or "file"
        p = os.path.join(up_dir, f"{len(saved) + len(skipped)}-{base}")
        with open(p, "wb") as o:
            shutil.copyfileobj(f.file, o)
        problem = pdftext.check(p) if p.lower().endswith(".pdf") else None
        if problem:  # tell the browser now, instead of a failed job later
            skipped.append({"file": base, "reason": problem})
            os.remove(p)
        else:
            saved.append(p)
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    path_list = [p.strip().strip('"') for p in paths.splitlines() if p.strip()]
    if not (saved or url_list or path_list):
        if skipped:
            err(" ".join(f"{s['file']}: {s['reason']}." for s in skipped))
        err("Add files, links or a folder path.")
    for u in url_list:
        try:
            core_net.check_url(u)
        except ValueError as e:
            err(str(e))
    if action == "learn":
        if not model:
            err("Choose a trained model to teach.")
        d = run_dir(model)
        cmd = [PY, "learn.py", "--run", d, "--passes", str(max(0.5, min(20, passes))),
               "--mix", str(max(0.0, min(10, mix)))] + (["--lr", str(lr)] if lr > 0 else [])
        log, run = os.path.join(d, "log.txt"), os.path.basename(d)
    else:
        cmd = [PY, "data/ingest.py", "--out", "data/raw/custom"]
        log, run = os.path.join(RUNS, "_data", "log.txt"), None
    if ocr == "off":
        cmd += ["--ocr", "off"]
    lang = re.sub(r"[^a-z_+]", "", (ocr_lang or "").lower())
    if lang and lang != "auto":
        cmd += ["--ocr-lang", lang]
    if path_list and delete_local:
        cmd.append("--delete-originals")
    if saved or path_list:
        cmd += ["--files", *saved, *path_list]
    if url_list:
        cmd += ["--urls", *url_list]
    out = _launch("texts", run or "add texts", cmd, log, run=run, cleanup=saved)
    return {**out, "skipped": skipped}


@app.post("/api/retrain")
async def retrain(files: list[UploadFile] = File(default=[]), model: str = Form(...), name: str = Form(...),
                  sources: str = Form(""), data_bin: bool = Form(False), urls: str = Form(""), paths: str = Form(""),
                  delete_local: bool = Form(False), steps: int = Form(0), lr: float = Form(0), mix: float = Form(0.5),
                  batch: int = Form(0), accum: int = Form(0), gguf: bool = Form(True),
                  ocr: str = Form("auto"), ocr_lang: str = Form("auto")):
    """Training → Retrain: a new model that starts from an existing one and learns other data (retrain.py).
    The original model is never changed."""
    src = run_dir(model)
    new = run_name(name, default="")
    if not new or new != name.strip():
        err("Give the new model a name: letters, digits, - _ . only.")
    out = os.path.join(RUNS, new)
    if out == src:
        err("Pick a new name: retraining keeps the original model as it is.")
    if any(os.path.exists(os.path.join(out, f)) for f in ("ckpt.pt", "adapter", "meta.json")):
        err(f"A model named {new} already exists. Pick another name.", 409)
    meta = _read_json(os.path.join(src, "meta.json")) or {}
    kind = meta.get("type")
    if kind not in ("scratch", "lora"):
        err(f"{model} can't be retrained (only from-scratch and fine-tuned models).")
    dirs = [corpus_dir(s) for s in sources.split(",") if s.strip()]
    if data_bin and kind != "scratch":
        err("The encoded corpus is for from-scratch models; pick text folders for a fine-tuned model.")
    bin_dir = os.path.join(ROOT, "data", "bin-code" if meta.get("task") == "code" else "bin")
    if data_bin and not os.path.exists(os.path.join(bin_dir, "train.bin")):
        err("There is no encoded corpus yet (Training → Data, step 3).")
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    path_list = [p.strip().strip('"') for p in paths.splitlines() if p.strip()]
    for u in url_list:
        try:
            core_net.check_url(u)
        except ValueError as e:
            err(str(e))
    up_dir = os.path.join(RUNS, "_uploads", str(int(time.time() * 1000)))
    saved = []
    for f in files:
        if not f.filename:
            continue
        os.makedirs(up_dir, exist_ok=True)
        base = os.path.basename(f.filename.replace("\\", "/")).strip().lstrip(".") or "file"
        p = os.path.join(up_dir, f"{len(saved)}-{base}")
        with open(p, "wb") as o:
            shutil.copyfileobj(f.file, o)
        saved.append(p)
    if not (dirs or data_bin or saved or url_list or path_list):
        err("Choose the new data: corpus folders, the encoded corpus, or your own files and links.")
    cmd = [PY, "retrain.py", "--from", src, "--out", out, "--mix", str(max(0.0, min(10, mix)))]
    if dirs:
        cmd += ["--source", *dirs]
    if data_bin:
        cmd += ["--data-bin", bin_dir]
    if steps > 0:
        cmd += ["--steps", str(min(steps, 10_000_000))]
    if lr > 0:
        cmd += ["--lr", str(lr)]
    if batch > 0:
        cmd += ["--batch", str(batch)]
    if accum > 0:
        cmd += ["--grad-accum", str(accum)]
    if gguf and kind == "lora":
        cmd.append("--gguf")
    if ocr == "off":
        cmd += ["--ocr", "off"]
    lang = re.sub(r"[^a-z_+]", "", (ocr_lang or "").lower())
    if lang and lang != "auto":
        cmd += ["--ocr-lang", lang]
    if saved or path_list:
        cmd += ["--files", *saved, *path_list]
        if path_list and delete_local:
            cmd.append("--delete-originals")
    if url_list:
        cmd += ["--urls", *url_list]
    os.makedirs(out, exist_ok=True)
    return _launch("training", new, cmd, os.path.join(out, "log.txt"), run=new, cleanup=saved)


@app.get("/api/texts/ocr")
def texts_ocr():
    """Whether scanned PDFs can be read here, for the hint under Your texts."""
    from data import pdftext

    if not pdftext.ocr_available():
        return {"available": False, "languages": [], "hint": pdftext.TESSERACT_HINT}
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = pdftext.tesseract_cmd()
    try:
        langs = sorted(l for l in pytesseract.get_languages(config="") if l not in ("osd", "equ", "snum"))
    except Exception:
        langs = []
    return {"available": True, "languages": langs, "auto": pdftext.ocr_languages("auto")}


@app.get("/api/runs")
def runs():
    return jobs.runs()


@app.get("/api/runs/{name}")
def run_detail(name: str):
    d = run_dir(name)
    rows = jobs.read_metrics(d)
    if len(rows) > 1500:  # keep charts light: thin train points, keep every eval point
        tr = [r for r in rows if r["kind"] == "train"]
        k = len(tr) // 1000 + 1
        keep = set(id(r) for r in tr[::k]) | set(id(r) for r in tr[-1:])
        rows = [r for r in rows if r["kind"] != "train" or id(r) in keep]
    return {"summary": jobs.run_summary(d), "metrics": rows, "log": jobs.tail(os.path.join(d, "log.txt"), 200)}


@app.get("/api/runs/{name}/export")
def export_run(name: str, data: bool = False):
    """Download the run as one .zip to continue training it on another PC (transfer.py)."""
    import transfer

    d = run_dir(name)
    if jobs.running() and jobs.JOB["run"] == name:
        err("Stop the training first (the last checkpoint is kept), then export.", 409)
    exp = os.path.join(RUNS, "_exports")
    os.makedirs(exp, exist_ok=True)
    for f in os.listdir(exp):  # keep only the newest export around; they can be large
        os.remove(os.path.join(exp, f))
    fn = f"{os.path.basename(d)}{'-with-data' if data else ''}.textllm.zip"
    out = transfer.export(d, os.path.join(exp, fn), with_data=data, log=lambda m: None)
    return FileResponse(out, filename=fn, media_type="application/zip")


@app.post("/api/runs/import")
async def import_run(file: UploadFile = File(...), name: str = Form(""), force: bool = Form(False)):
    import transfer

    tmp = os.path.join(RUNS, "_uploads", f"import-{int(time.time() * 1000)}.zip")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        msgs = []
        try:
            got = transfer.import_bundle(tmp, name.strip() or None, force, log=msgs.append)
        except SystemExit as e:
            err(str(e), 409 if "exists" in str(e) else 400)
        except Exception as e:  # noqa: BLE001  (bad zip, disk full, …)
            err(f"Import failed: {e}", 400)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    ENGINES.clear()
    return {"ok": True, "name": got, "log": "\n".join(msgs)}


@app.delete("/api/runs/{name}")
def delete_run(name: str):
    d = run_dir(name)
    if jobs.running() and jobs.JOB["run"] == name:
        err("Stop the job first.", 409)
    shutil.rmtree(d)
    ENGINES.clear()
    return {"ok": True}


# ---------------------------------------------------------------- chat
ENGINES = {}
_engine_lock = threading.Lock()


def engine(name):
    models = {os.path.basename(v): v for v in list_models().values()}
    if name not in models:
        err(f"Model {name} isn't trained yet.", 404)
    d = models[name]
    stamp = weights_stamp(d)  # changes after (feedback) training or a new GGUF copy
    with _engine_lock:
        cached = ENGINES.get(name)
        if not cached or cached[0] != stamp:  # reload after (feedback) training
            ENGINES.clear()
            ENGINES[name] = (stamp, Engine(d))
        return ENGINES[name][1]


@app.get("/api/models")
def models(request: Request):
    out, default = [], db.get_settings().get("default_model")
    for label, d in list_models().items():
        s = jobs.run_summary(d)
        stats = db.q("SELECT COUNT(rating) n, AVG(rating) avg FROM messages WHERE model=? AND rating IS NOT NULL",
                     (s["name"],), one=True)
        meta_p = os.path.join(d, "meta.json")
        style = (json.load(open(meta_p)).get("style") if os.path.exists(meta_p) else None)
        out.append({**s, "label": label, "style": style, "ratings": stats["n"], "avg_rating": stats["avg"],
                    "pending_feedback": db.pending_feedback(s["name"]), "default": s["name"] == default,
                    "fast": bool(gguf_path(d))})
    out.sort(key=lambda m: not m["default"])
    if not is_admin(request):  # chat only needs to pick a model
        keep = ("name", "type", "label", "level", "ratings", "avg_rating", "default", "style", "task")
        out = [{k: m.get(k) for k in keep} for m in out]
    return out


@app.get("/api/conversations")
def conversations(request: Request):
    """The signed-in user's own chats."""
    return db.q("SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) n "
                "FROM conversations c WHERE c.user_id=? ORDER BY updated DESC LIMIT 200", (me(request)["id"],))


@app.get("/api/conversations/{cid}")
def conversation(cid: int, request: Request):
    c = own_conversation(request, cid)
    return {**c, "messages": db.q("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)),
            "mine": c["user_id"] == me(request)["id"]}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: int, request: Request):
    own_conversation(request, cid)
    db.ex("DELETE FROM messages WHERE conversation_id=?", (cid,))
    db.ex("DELETE FROM conversations WHERE id=?", (cid,))
    return {"ok": True}


class ChatReq(BaseModel):
    model: str
    message: str = ""
    conversation_id: int | None = None
    regenerate: int | None = None     # user message id to answer again
    format: str = "free"              # writing/formats.json id
    fields: dict | None = None        # the format's form (tone, audience, ...)
    temperature: float | None = None  # None = the format's default
    max_tokens: int | None = None
    # anti-repetition; None = the format's / model type's default (chat_engine.REPEAT_DEFAULTS)
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    no_repeat_ngram: int | None = None


@app.get("/api/formats")
def formats_list():
    return F.public_list()


class CompleteReq(BaseModel):
    model: str
    prefix: str
    suffix: str = ""
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95


@app.post("/api/code/complete")
def code_complete(r: CompleteReq):
    """Editor-style completion: the code that goes between prefix and suffix (fill-in-the-middle).
    Any signed-in user may call it, e.g. from an editor plugin that sends the Studio's session cookie."""
    eng = engine(r.model)
    out = ""
    with eng.lock:
        for out in eng.code_complete(r.prefix[-24000:], r.suffix[:8000], max(1, min(1024, r.max_tokens)),
                                     max(0.0, min(2.0, r.temperature)), max(0.05, min(1.0, r.top_p))):
            pass
    return {"completion": out, "model": r.model}


@app.post("/api/chat")
def chat(r: ChatReq, request: Request):
    eng = engine(r.model)
    if r.regenerate:
        um = db.q("SELECT * FROM messages WHERE id=? AND role='user'", (r.regenerate,), one=True) or err("not found", 404)
        own_conversation(request, um["conversation_id"])
        cid, uid, text = um["conversation_id"], um["id"], um["content"]
        fmt = um.get("format") or "free"
        fields = json.loads(um["fields"]) if um.get("fields") else None
    else:
        text = r.message.strip() or err("Empty message")
        if r.conversation_id:
            own_conversation(request, r.conversation_id)
        fmt = r.format if r.format in F.formats() else "free"
        fields = {k: str(v)[:300] for k, v in (r.fields or {}).items() if str(v).strip()} or None
        cid = r.conversation_id or db.new_conversation(text, r.model, me(request)["id"])
        uid = db.add_message(cid, "user", text, fmt=fmt, fields=fields)
    hist = db.history(cid, before_id=uid)
    samp = {"temperature": r.temperature, "max_tokens": r.max_tokens,
            "repetition_penalty": None if r.repetition_penalty is None else max(1.0, min(2.0, r.repetition_penalty)),
            "frequency_penalty": None if r.frequency_penalty is None else max(0.0, min(2.0, r.frequency_penalty)),
            "no_repeat_ngram": None if r.no_repeat_ngram is None else max(0, min(10, r.no_repeat_ngram))}
    model_text, sources = text, None
    if F.get(fmt).get("context") == "news":  # "Write about today's news": recent articles go into the prompt
        model_text, sources = with_news(text, fields)

    def gen():
        yield json.dumps({"type": "start", "conversation_id": cid, "user_id": uid, "format": fmt}) + "\n"
        if sources is not None:
            yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        full, score = "", None
        try:
            for ev in eng.write(fmt, model_text, fields, hist, samp):
                if ev["type"] == "text":
                    full = ev["text"]
                    yield json.dumps({"type": "text", "text": full}) + "\n"
                elif ev["type"] == "final":
                    full, score = ev["text"], ev["check"]["score"]
                    yield json.dumps({"type": "check", "check": ev["check"]}) + "\n"
                else:  # status, outline
                    yield json.dumps(ev) + "\n"
        except Exception as e:  # report instead of breaking the stream
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        finally:  # also runs if the browser stops the stream: keep what was written
            mid = db.add_message(cid, "assistant", full.strip() or "…", model=r.model, prompt_id=uid,
                                 fmt=fmt, check_score=score, fields={"sources": sources} if sources else None)
        yield json.dumps({"type": "done", "message_id": mid}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def with_news(text, fields):
    """Append the most relevant recent crawled articles to the request. Returns (prompt, sources)."""
    days = 7
    try:
        days = max(1, min(90, int(re.sub(r"[^\d]", "", str((fields or {}).get("days", "")) or "7") or 7)))
    except ValueError:
        pass
    arts = crawl.recent_articles(text, n=5, days=days)
    if not arts:
        return text + "\n\n(No recent articles have been collected yet. Say so briefly and write what you can.)", []
    lines = [text.strip(), "", f"Recent articles (last {days} days):"]
    for i, a in enumerate(arts, 1):
        when = (a.get("published") or "")[:10] or time.strftime("%Y-%m-%d", time.localtime(a["fetched"]))
        lines.append(f"{i}. {a['title'] or a['url']} ({a['source_name']}, {when})\n{a['excerpt'][:900].strip()}")
    return "\n".join(lines), [{"title": a["title"] or a["url"], "url": a["url"], "source": a["source_name"]} for a in arts]


# ---------------------------------------------------------------- feedback
class RateReq(BaseModel):
    rating: int | None = None
    comment: str | None = None
    correction: str | None = None


@app.post("/api/messages/{mid}/rate")
def rate(mid: int, r: RateReq, request: Request):
    if r.rating is not None and not 1 <= r.rating <= 5:
        err("rating must be 1–5")
    m = db.q("SELECT conversation_id FROM messages WHERE id=?", (mid,), one=True) or err("not found", 404)
    own_conversation(request, m["conversation_id"])
    m = db.rate(mid, r.rating, r.comment, r.correction) or err("not found", 404)
    auto = maybe_auto_train(m["model"])
    return {"message": m, "auto_training": auto,
            "pending": db.pending_feedback(m["model"]) if is_admin(request) else None}


def maybe_auto_train(model):
    s = db.get_settings()
    if not s["auto_train"] or not model or jobs.running():
        return False
    if db.pending_feedback(model) < int(s["auto_threshold"]):
        return False
    try:
        d = os.path.join(RUNS, model)
        cmd = feedback_cmd(d, s) + (["--gate"] if s.get("gate_on_feedback", True) else [])
        jobs.launch("feedback", model, cmd, os.path.join(d, "log.txt"), run=model)
        return True
    except RuntimeError:
        return False


@app.get("/api/feedback")
def feedback_list(model: str = "", only: str = "all"):
    sql = ("SELECT a.*, u.content prompt, c.title, COALESCE(us.display_name, us.username) author "
           "FROM messages a JOIN messages u ON u.id=a.prompt_id "
           "JOIN conversations c ON c.id=a.conversation_id LEFT JOIN users us ON us.id=c.user_id "
           "WHERE a.role='assistant' "
           "AND (a.rating IS NOT NULL OR a.correction IS NOT NULL)")
    args = []
    if model:
        sql += " AND a.model=?"
        args.append(model)
    if only == "pending":
        sql += " AND a.trained_in IS NULL"
    return db.q(sql + " ORDER BY a.rated_at DESC LIMIT 500", args)


class FeedbackTrainReq(BaseModel):
    model: str
    epochs: float | None = None   # None = the saved settings (Feedback → Training options)


def feedback_cmd(d, s, epochs=None):
    return [PY, "feedback_train.py", "--run", d, "--epochs", str(epochs or s.get("feedback_epochs", 3)),
            "--lr", str(s.get("feedback_lr", 5e-5)), "--beta", str(s.get("feedback_beta", 0.1))]


@app.post("/api/feedback/train")
def feedback_train(r: FeedbackTrainReq):
    d = run_dir(r.model)
    if db.pending_feedback(r.model) == 0:
        err("No new ratings for this model yet.")
    return _launch("feedback", r.model, feedback_cmd(d, db.get_settings(), r.epochs),
                   os.path.join(d, "log.txt"), run=r.model)


class SettingsReq(BaseModel):
    device: str | None = None
    allow_private_urls: bool | None = None   # read links that point into the local network (NAS, intranet)
    auto_train: bool | None = None
    auto_threshold: int | None = None
    gate_on_feedback: bool | None = None
    feedback_epochs: float | None = None
    feedback_lr: float | None = None
    feedback_beta: float | None = None


@app.post("/api/settings")
def settings(r: SettingsReq):
    kw = {k: v for k, v in r.model_dump().items() if v is not None}
    if "device" in kw:
        kw["device"] = kw["device"].strip().lower() or "auto"
        if not re.fullmatch(r"auto|cpu|mps|(xpu|cuda)(:\d)?", kw["device"]):
            err("device must be auto, cpu, cuda, cuda:N, mps, xpu or xpu:N")
    for k, lo, hi in (("feedback_epochs", 0.5, 20), ("feedback_lr", 1e-7, 1e-2), ("feedback_beta", 0.01, 2)):
        if k in kw:
            kw[k] = max(lo, min(hi, kw[k]))
    out = db.set_settings(**kw)
    if "device" in kw:
        apply_device(kw["device"])
    if "allow_private_urls" in kw:
        apply_private_urls(kw["allow_private_urls"])
    return out


# ---------------------------------------------------------------- corpus
RAW = os.path.join(ROOT, "data", "raw")
EOT = "<|endoftext|>"
SOURCE_NOTES = {
    "gutenberg": "Project Gutenberg books (public domain)", "fineweb": "FineWeb-Edu web text (ODC-By)",
    "custom": "Your own files and links", "news": "News & web crawler", "pg19": "PG-19 books (Apache 2.0)",
    "standard_ebooks": "Standard Ebooks (CC0)", "common_corpus": "Common Corpus, Open Culture (public domain)",
    "archive": "Internet Archive books (public domain)", "cosmopedia": "Cosmopedia (Apache 2.0)",
    "wikipedia": "Wikipedia (CC BY-SA 4.0)", "common_pile": "Common Pile, openly licensed writing (CC BY / BY-SA / PD)",
}


def corpus_dir(source):
    """data/raw/<source> (one folder level, or news/<id>); refuses anything outside data/raw."""
    d = core_paths.under(RAW, *source.strip().replace("\\", "/").split("/"))
    if d is None or not os.path.isdir(d):
        err(f"no corpus folder named {source}", 404)
    return d


def _shards(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))


def _count_docs(files, limit_bytes=200_000_000):
    """Documents (EOT-separated) in the shards; big folders are sampled and scaled."""
    total = sum(os.path.getsize(f) for f in files)
    docs, read = 0, 0
    for f in files:
        with open(f, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                docs += chunk.count(EOT.encode())
                read += len(chunk)
        if read > limit_bytes:
            return int(docs * total / read), True
    return docs, False


@app.get("/api/corpus")
def corpus_list():
    out = []
    if not os.path.isdir(RAW):
        return {"sources": out, "total_mb": 0}
    for name in sorted(os.listdir(RAW)):
        d = os.path.join(RAW, name)
        if not os.path.isdir(d):
            continue
        groups = [(name, d)]
        if name == "news":  # one row per crawler source
            groups = [(f"news/{s}", os.path.join(d, s)) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
        for key, folder in groups:
            files = _shards(folder)
            size = sum(os.path.getsize(f) for f in files)
            docs, approx = _count_docs(files) if files else (0, False)
            out.append({"source": key, "note": SOURCE_NOTES.get(name, ""), "files": len(files), "mb": round(size / 1e6, 1),
                        "docs": docs, "docs_approx": approx, "tokens_est": size // 4,
                        "updated": max((os.path.getmtime(f) for f in files), default=os.path.getmtime(folder))})
    return {"sources": out, "total_mb": round(sum(s["mb"] for s in out), 1),
            "encoded": jobs.data_status()["train_tokens"]}


@app.get("/api/corpus/sample")
def corpus_sample(source: str, n: int = 3):
    """A few random documents (their first 700 characters) from a corpus folder."""
    import random

    files = _shards(corpus_dir(source))
    if not files:
        return {"samples": []}
    samples = []
    rng = random.Random()
    for _ in range(max(1, min(10, n)) * 4):
        f = rng.choice(files)
        size = os.path.getsize(f)
        with open(f, "rb") as fh:
            fh.seek(rng.randrange(0, max(1, size - 50_000)) if size > 60_000 else 0)
            blob = fh.read(60_000).decode("utf-8", "replace")
        parts = [p.strip() for p in blob.split(EOT) if p.strip()]
        if size > 60_000:
            parts = parts[1:-1]  # the first and last pieces are cut mid-document
        if parts:
            doc = rng.choice(parts)
            samples.append({"file": os.path.relpath(f, RAW).replace(os.sep, "/"), "text": doc[:700], "chars": len(doc)})
        if len(samples) >= n:
            break
    return {"samples": samples}


@app.delete("/api/corpus/{source:path}")
def corpus_delete(source: str):
    """Remove a corpus folder. The tokenizer and encoded data then need re-running."""
    if jobs.running() and jobs.JOB["kind"] in ("data", "training", "texts", "crawl"):
        err("Wait for the running job to finish first.", 409)
    d = corpus_dir(source)
    shutil.rmtree(d)
    if source.startswith("news/"):  # the crawler would otherwise never fetch these again: forget them
        try:
            with crawl.db() as c:
                c.execute("DELETE FROM hashes WHERE url IN (SELECT url FROM items WHERE source=?)", (source[5:],))
                c.execute("DELETE FROM items WHERE source=?", (source[5:],))
        except Exception:
            pass
    return {"ok": True, "note": "Re-run Build the tokenizer and Encode the corpus so the change reaches training."}


# ---------------------------------------------------------------- news & web crawler
CRAWL_LOG = os.path.join(RUNS, "_crawler", "log.txt")


def crawl_cmd(s):
    m = s.get("crawl_learn_model")
    if m and not os.path.isdir(os.path.join(RUNS, m)):
        m = None
    cmd = crawl.crawl_command(os.path.join(RUNS, m) if m else None, s["crawl_min_docs"], s["crawl_passes"],
                              s.get("crawl_languages", ""), s.get("crawl_gate", True))
    cmd[0] = PY
    if m and s.get("crawl_max_drop", 5) != 5:
        cmd += ["--max-drop", str(s["crawl_max_drop"])]
    return cmd, (m or None)


def crawl_next(s):
    """When the in-app scheduler runs next; None when off or when the OS scheduler does it."""
    if not s["crawl_enabled"] or crawl.schedule_status()["installed"]:
        return None
    return s["crawl_last_start"] + float(s["crawl_every_hours"]) * 3600


def start_crawl(s):
    cmd, run = crawl_cmd(s)
    st = jobs.launch("crawl", "news & web", cmd, CRAWL_LOG, run=run)
    db.set_settings(crawl_last_start=time.time())
    return st


def crawl_scheduler():
    """Runs a crawl when one is due. If another job is busy it waits for it to finish."""
    while True:
        try:
            s = db.get_settings()
            nxt = crawl_next(s)
            if nxt is not None and time.time() >= nxt and not jobs.running():
                start_crawl(s)
        except Exception as e:  # keep the scheduler alive whatever happens
            print("crawler scheduler:", e, flush=True)
        time.sleep(60)


@app.on_event("startup")
def _start_scheduler():
    apply_private_urls(db.get_settings().get("allow_private_urls", False))
    apply_device(db.get_settings().get("device", "auto"))
    threading.Thread(target=crawl_scheduler, daemon=True, name="crawl-scheduler").start()
    threading.Thread(target=copilot.auto_scheduler, daemon=True, name="code-updates-scheduler").start()
    # The server answers at once; PyTorch (seconds to import) loads meanwhile for the device info.
    threading.Thread(target=device_info, daemon=True, name="warm-up").start()


@app.on_event("shutdown")
def _stop_job():
    """Closing the Studio (Ctrl+C) stops the running job too; it keeps its last checkpoint."""
    if jobs.running():
        print("stopping the running job…", flush=True)
        jobs.stop(wait=True)


@app.get("/api/crawler")
def crawler_state():
    s = db.get_settings()
    return {"sources": crawl.load_sources(), "licenses": crawl.LICENSES, "stats": crawl.stats(),
            "settings": {k: v for k, v in s.items() if k.startswith("crawl_")} | {"next": crawl_next(s)},
            "language_names": crawl.langmod.NAMES, "log": jobs.tail(CRAWL_LOG, 150)}


class SourceReq(BaseModel):
    url: str | None = None
    name: str | None = None
    kind: str | None = None          # feed | page | links | mastodon | bluesky
    license: str | None = None
    rights_confirmed: bool | None = None
    terms_confirmed: bool | None = None   # social sources: the server's terms allow it
    min_chars: int | None = None          # social sources: skip shorter posts
    enabled: bool | None = None
    max_items: int | None = None
    credit: str | None = None


@app.post("/api/crawler/sources")
def crawler_add(r: SourceReq):
    try:
        return crawl.add_source(r.model_dump(exclude_none=True))
    except ValueError as e:
        err(str(e))


@app.patch("/api/crawler/sources/{sid}")
def crawler_edit(sid: str, r: SourceReq):
    try:
        return crawl.update_source(sid, r.model_dump(exclude_none=True))
    except LookupError as e:
        err(str(e), 404)
    except ValueError as e:
        err(str(e))


@app.delete("/api/crawler/sources/{sid}")
def crawler_delete(sid: str):
    try:
        crawl.delete_source(sid)
    except LookupError as e:
        err(str(e), 404)
    return {"ok": True}


class CrawlSettingsReq(BaseModel):
    enabled: bool | None = None
    every_hours: float | None = None
    learn_model: str | None = None   # "" = only add to the corpus
    min_docs: int | None = None
    passes: float | None = None
    languages: str | None = None     # "en,hy"; "" = keep every language
    gate: bool | None = None         # quick check after learning, undo if worse
    max_drop: float | None = None


@app.post("/api/crawler/settings")
def crawler_settings(r: CrawlSettingsReq):
    kw = {f"crawl_{k}": v for k, v in r.model_dump(exclude_none=True).items()}
    if "crawl_every_hours" in kw:
        kw["crawl_every_hours"] = max(0.25, min(24 * 7, kw["crawl_every_hours"]))
    if kw.get("crawl_learn_model"):
        run_dir(kw["crawl_learn_model"])
    if "crawl_min_docs" in kw:
        kw["crawl_min_docs"] = max(1, kw["crawl_min_docs"])
    if "crawl_passes" in kw:
        kw["crawl_passes"] = max(0.5, min(10, kw["crawl_passes"]))
    if "crawl_languages" in kw:
        kw["crawl_languages"] = ",".join(sorted({l.strip().lower() for l in re.split(r"[,\s]+", kw["crawl_languages"])
                                                 if re.fullmatch(r"[a-z]{2,3}", l.strip().lower())}))
    if "crawl_max_drop" in kw:
        kw["crawl_max_drop"] = max(0, min(100, kw["crawl_max_drop"]))
    db.set_settings(**kw)
    return crawler_state()


@app.post("/api/crawler/schedule")
def crawler_os_schedule(install: bool = True):
    """Register the crawl with Windows Task Scheduler / cron (runs when the Studio is closed), or remove it."""
    s = db.get_settings()
    try:
        if install:
            m = s.get("crawl_learn_model")
            msg = crawl.install_schedule(s["crawl_every_hours"], os.path.join(RUNS, m) if m and os.path.isdir(os.path.join(RUNS, m)) else None,
                                         s["crawl_min_docs"], s["crawl_passes"], s.get("crawl_languages", ""),
                                         s.get("crawl_gate", True))
        else:
            msg = crawl.uninstall_schedule()
    except FileNotFoundError:
        err("No scheduler found on this computer (schtasks on Windows, crontab on Linux/macOS).")
    except RuntimeError as e:
        err(f"Could not change the schedule: {e}")
    return {"message": msg, **crawler_state()}


@app.post("/api/crawler/run")
def crawler_run():
    try:
        return start_crawl(db.get_settings())
    except RuntimeError as e:
        err(str(e), 409)


# ---------------------------------------------------------------- analytics
@app.get("/api/analytics")
def analytics():
    rs = jobs.runs()
    dist = {r["rating"]: r["n"] for r in db.q(
        "SELECT rating, COUNT(*) n FROM messages WHERE rating IS NOT NULL GROUP BY rating")}
    daily = db.q("SELECT date(rated_at, 'unixepoch', 'localtime') day, COUNT(*) n, AVG(rating) avg "
                 "FROM messages WHERE rating IS NOT NULL GROUP BY day ORDER BY day DESC LIMIT 30")[::-1]
    per_model = db.q("SELECT model, COUNT(*) answers, COUNT(rating) rated, AVG(rating) avg, "
                     "SUM(CASE WHEN rating>=4 THEN 1 ELSE 0 END) liked, "
                     "SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) disliked, "
                     "SUM(CASE WHEN correction IS NOT NULL THEN 1 ELSE 0 END) corrected "
                     "FROM messages WHERE role='assistant' GROUP BY model")
    # Ratings before vs after each feedback training, to show whether it helped.
    trend = db.q("SELECT model, trained_in, AVG(rating) avg, COUNT(*) n FROM messages "
                 "WHERE rating IS NOT NULL GROUP BY model, trained_in")
    per_format = db.q("SELECT COALESCE(format, 'free') format, COUNT(*) answers, COUNT(rating) rated, "
                      "AVG(rating) avg, AVG(check_score) checks, "
                      "SUM(CASE WHEN correction IS NOT NULL THEN 1 ELSE 0 END) corrected "
                      "FROM messages WHERE role='assistant' GROUP BY COALESCE(format, 'free')")
    counts = db.q("SELECT (SELECT COUNT(*) FROM conversations) chats, "
                  "(SELECT COUNT(*) FROM messages WHERE role='assistant') answers, "
                  "(SELECT COUNT(rating) FROM messages) rated, (SELECT AVG(rating) FROM messages) avg_rating",
                  one=True)
    curves = {}
    for r in rs[:8]:
        rows = jobs.read_metrics(os.path.join(RUNS, r["name"]))
        curves[r["name"]] = [{"step": x["step"], "loss": x["loss"]} for x in rows if x["kind"] == "val"]
    return {
        "runs": rs, "counts": counts, "rating_distribution": dist, "daily": daily, "per_model": per_model,
        "trend": trend, "val_curves": curves, "per_format": per_format, "jobs": jobs.history(30),
        "train_hours": sum(r["train_seconds"] for r in rs) / 3600,
        "pending_feedback": db.pending_feedback(),
    }


# ---------------------------------------------------------------- evaluation, A/B votes, promotion
class EvalReq(BaseModel):
    model: str
    quick: bool = False
    max_tokens: int = 0
    formats: str = ""      # comma-separated format ids, "" = all
    limit: int = 0         # at most this many prompts, 0 = all


@app.post("/api/eval")
def eval_start(r: EvalReq):
    d = run_dir(r.model)
    cmd = [PY, "evaluate.py", "--run", d] + (["--quick"] if r.quick else []) + \
          (["--max-tokens", str(r.max_tokens)] if r.max_tokens else []) + \
          (["--formats", re.sub(r"[^a-z0-9_,]", "", r.formats)] if r.formats.strip() else []) + \
          (["--limit", str(r.limit)] if r.limit > 0 else [])
    return _launch("eval", r.model, cmd, os.path.join(d, "log.txt"), run=r.model)


@app.get("/api/eval")
def eval_overview():
    out = []
    for m in jobs.runs():
        h = os.path.join(RUNS, m["name"], "eval", "history.jsonl")
        if os.path.exists(h):
            rows = [r for r in (json.loads(l) for l in open(h) if l.strip()) if not r.get("partial") and not r.get("rejected")]
            if rows:
                out.append({"model": m["name"], "latest": rows[-1], "history": rows[-20:]})
    s = db.get_settings()
    return {"evals": out, "pairs": db.pair_stats(), "default_model": s.get("default_model"),
            "formats": {f["id"]: f["label"] for f in F.public_list()}}


@app.get("/api/eval/{name}/latest")
def eval_latest(name: str):
    d = run_dir(name)
    h = os.path.join(d, "eval", "history.jsonl")
    if not os.path.exists(h):
        err("No evaluation yet.", 404)
    last = [json.loads(l) for l in open(h) if l.strip()][-1]
    return json.load(open(os.path.join(d, "eval", last["file"])))


class PairsReq(BaseModel):
    candidate: str
    current: str
    judge: str = ""
    quick: bool = False


@app.post("/api/pairs")
def pairs_make(r: PairsReq):
    a, b = run_dir(r.candidate), run_dir(r.current)
    if a == b:
        err("Pick two different models.")
    cmd = [PY, "evaluate.py", "--pairs", a, b] + (["--judge", r.judge] if r.judge.strip() else []) + \
          (["--quick"] if r.quick else [])
    return _launch("eval", f"{r.candidate} vs {r.current}", cmd, os.path.join(RUNS, "_data", "log.txt"))


def _flip(pid):  # which side the candidate is shown on; stable per pair, not obvious to voters
    return (pid * 2654435761 >> 7) % 2 == 1


@app.get("/api/pairs/next")
def pairs_next():
    p = db.next_pair()
    if not p:
        return {"pair": None, "left": db.q("SELECT COUNT(*) n FROM pairs WHERE vote IS NULL", one=True)["n"]}
    fl = _flip(p["id"])
    return {"pair": {"id": p["id"], "format": p["format"], "prompt": p["prompt"],
                     "left": p["text_b"] if fl else p["text_a"], "right": p["text_a"] if fl else p["text_b"]},
            "left": db.q("SELECT COUNT(*) n FROM pairs WHERE vote IS NULL", one=True)["n"]}


class VoteReq(BaseModel):
    choice: str  # left | right | tie


@app.post("/api/pairs/{pid}/vote")
def pairs_vote(pid: int, r: VoteReq, request: Request):
    if r.choice not in ("left", "right", "tie"):
        err("choice must be left, right or tie")
    fl = _flip(pid)
    v = "tie" if r.choice == "tie" else ("a" if (r.choice == "left") != fl else "b")
    db.vote_pair(pid, v, me(request)["username"])
    return pairs_next()


@app.get("/api/gate")
def gate_check(candidate: str, current: str = ""):
    run_dir(candidate)
    cur = current or db.get_settings().get("default_model") or ""
    return {"candidate": candidate, "current": cur or None,
            **gate.decide(candidate, cur if cur and cur != candidate else None)}


class PromoteReq(BaseModel):
    model: str
    force: bool = False


@app.post("/api/promote")
def promote(r: PromoteReq):
    run_dir(r.model)
    cur = db.get_settings().get("default_model")
    g = gate.decide(r.model, cur if cur and cur != r.model else None)
    if not g["pass"] and not r.force:
        err("The gate did not pass: " + " ".join(g["reasons"]), 409)
    db.set_settings(default_model=r.model)
    return {"default_model": r.model, "gate": g}


@app.post("/api/runs/{name}/gguf")
def make_gguf(name: str, quant: str = "Q4_K_M"):
    d = run_dir(name)
    if quant not in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"):
        err("quant must be Q4_K_M, Q5_K_M, Q6_K or Q8_0")
    return _launch("export", name, [PY, "export_gguf.py", "--run", d, "--quant", quant],
                   os.path.join(d, "log.txt"), run=name)


# ---------------------------------------------------------------- web
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    print(f"Text LLM Studio → http://{a.host}:{a.port}")
    if not auth.count_users():
        print("No accounts yet: open the link to create the admin account "
              "(or run: python -m studio.auth add NAME --admin).")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
