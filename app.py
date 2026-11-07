"""Text LLM Studio: web app for data, training, analytics, chat and feedback.

    pip install -r requirements.txt
    python app.py                 # then open http://127.0.0.1:7860
    python app.py --port 8000 --host 0.0.0.0

Sign-in is required. The first visit creates the admin account; admins add more people on the
Users page (or with `python -m studio.auth add NAME`). Regular users only see Chat and Settings.
"""
import argparse, glob, json, os, re, shutil, threading, time

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

PY = jobs.PY
WEB = os.path.join(ROOT, "web")
app = FastAPI(title="Text LLM Studio")


def err(msg, code=400):
    raise HTTPException(code, msg)


def _launch(*a, **kw):
    try:
        return jobs.launch(*a, **kw)
    except RuntimeError as e:
        err(str(e), 409)


def run_dir(name):
    d = os.path.join(RUNS, os.path.basename(name))
    if not os.path.isdir(d):
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
]]


@app.middleware("http")
async def require_login(request: Request, call_next):
    path, method = request.url.path, request.method
    if not path.startswith("/api/") or (method, path) in PUBLIC_ROUTES:
        return await call_next(request)
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


def _set_cookie(resp, token):
    resp.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400, httponly=True, samesite="lax")


class LoginReq(BaseModel):
    username: str
    password: str
    display_name: str | None = None


@app.get("/api/auth/state")
def auth_state(request: Request):
    return {"needs_setup": auth.count_users() == 0,
            "user": auth.user_for_token(request.cookies.get(auth.COOKIE))}


@app.post("/api/auth/setup")
def auth_setup(r: LoginReq, response: Response):
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
    _set_cookie(response, token)
    return user


@app.post("/api/auth/login")
def auth_login(r: LoginReq, response: Response):
    token, user = auth.login(r.username, r.password)
    if not token:
        err("Wrong username or password.", 401)
    _set_cookie(response, token)
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
        _set_cookie(response, token)
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
        _DEVICE = devices.info()  # doesn't change while the app runs
    return _DEVICE


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
    step: str            # download | tokenizer | encode
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
    vocab: int = 8192
    hf_sets: bool = True


@app.post("/api/data")
def data(r: DataReq):
    cmds = {
        "download": [PY, "data/build_corpus.py", "--gutenberg-max", str(r.books), "--fineweb-gb", str(r.web_gb),
                     "--gutenberg-method", r.gutenberg_method if r.gutenberg_method in ("mirror", "harvest") else "mirror",
                     "--pg19-books", str(max(0, r.pg19_books)), "--standard-ebooks", str(max(0, r.standard_ebooks)),
                     "--common-corpus-gb", str(max(0.0, r.common_corpus_gb)),
                     "--archive-books", str(max(0, r.archive_books)),
                     "--cosmopedia-gb", str(max(0.0, r.cosmopedia_gb)),
                     "--cosmopedia-subsets", re.sub(r"[^a-z0-9_,]", "", r.cosmopedia_subsets) or "stories"],
        "tokenizer": [PY, "tokenizer/train_tokenizer.py", "--vocab", str(r.vocab)],
        "encode": [PY, "tokenizer/train_tokenizer.py", "--encode"],
        # writing instructions for fine-tuning: posts, stories, essays (rights-cleared sources only)
        "instructions": [PY, "data/build_instructions.py", "--from-studio"] + (["--hf-sets"] if r.hf_sets else []),
    }
    if r.step not in cmds:
        err("unknown step")
    return _launch("data", r.step, cmds[r.step], os.path.join(RUNS, "_data", "log.txt"))


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


@app.post("/api/train")
def train(r: TrainReq):
    name = "".join(c if c.isalnum() or c in "-_." else "-" for c in r.name.strip()) or "my-model"
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    if r.mode == "scratch":
        if not (r.resume and os.path.exists(os.path.join(out, "train_cmd.json"))) and \
                not os.path.exists(os.path.join(ROOT, "data/bin/train.bin")):
            err("Prepare the training data first (Training → Data, steps 1–3).")
        tok_copy = os.path.join(out, "tokenizer.json")
        if not (r.resume and os.path.exists(tok_copy)):
            # Keep the tokenizer with the weights, so the run folder can be moved to another PC.
            src = os.path.join(ROOT, "tokenizer/bpe.json")
            if not os.path.exists(src):
                err("Train the tokenizer first (Training → Data, step 2).")
            shutil.copyfile(src, tok_copy)
        meta_p = os.path.join(out, "meta.json")
        if not (r.resume and os.path.exists(meta_p)):
            json.dump({"type": "scratch", "tok": "tokenizer.json", "preset": r.preset}, open(meta_p, "w"))
        cmd = [PY, "train.py", "--out", out, "--preset", r.preset, "--vocab", str(r.vocab),
               "--max-steps", str(r.steps), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--warmup", str(max(10, r.steps // 40)), "--eval-every", str(max(10, r.steps // 80))]
        if r.resume:
            cmd.append("--resume")
    else:
        import glob
        inst = os.path.join(ROOT, "data/instructions/train.jsonl")
        use_inst = r.instructions and os.path.exists(inst)
        resuming_saved = r.resume and os.path.exists(os.path.join(out, "train_cmd.json"))
        if not resuming_saved and not use_inst and not glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True):
            err("Add some text first: build writing instructions or download books (Training → Data).")
        cmd = [PY, "finetune/finetune_lora.py", "--out", out, "--base", r.base, "--examples", str(r.examples),
               "--seq", str(r.seq), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--max-steps", "0" if r.epochs > 0 else str(r.steps), "--eval-every", str(max(5, r.steps // 20)),
               "--save-every", str(max(5, r.steps // 10)), "--lora-r", str(r.lora_r)]
        if r.epochs > 0:
            cmd += ["--epochs", str(r.epochs)]
        if use_inst:
            cmd += ["--instructions", inst, "--book-share", str(r.book_share)]
        if r.gguf:
            cmd += ["--gguf"]
        if r.resume:
            cmd.append("--resume")
    saved_cmd = os.path.join(out, "train_cmd.json")
    if r.resume and os.path.exists(saved_cmd):
        # Continue with the run's own settings (LR schedule, batch, data), not today's form values.
        cmd = [PY] + json.load(open(saved_cmd))
        cmd = [out if i > 0 and cmd[i - 1] == "--out" else c for i, c in enumerate(cmd)]
        if "--resume" not in cmd:
            cmd.append("--resume")
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
                action: str = Form("learn"), model: str = Form(""), passes: int = Form(3),
                delete_local: bool = Form(False)):
    up_dir = os.path.join(RUNS, "_uploads", str(int(time.time() * 1000)))
    saved = []
    for f in files:
        if not f.filename:
            continue
        os.makedirs(up_dir, exist_ok=True)
        p = os.path.join(up_dir, os.path.basename(f.filename))
        with open(p, "wb") as o:
            shutil.copyfileobj(f.file, o)
        saved.append(p)
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    path_list = [p.strip().strip('"') for p in paths.splitlines() if p.strip()]
    if not (saved or url_list or path_list):
        err("Add files, links or a folder path.")
    if action == "learn":
        if not model:
            err("Choose a trained model to teach.")
        d = run_dir(model)
        cmd = [PY, "learn.py", "--run", d, "--passes", str(passes)]
        log, run = os.path.join(d, "log.txt"), os.path.basename(d)
    else:
        cmd = [PY, "data/ingest.py", "--out", "data/raw/custom"]
        log, run = os.path.join(RUNS, "_data", "log.txt"), None
    if path_list and delete_local:
        cmd.append("--delete-originals")
    if saved or path_list:
        cmd += ["--files", *saved, *path_list]
    if url_list:
        cmd += ["--urls", *url_list]
    return _launch("texts", run or "add texts", cmd, log, run=run, cleanup=saved)


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
        keep = ("name", "type", "label", "level", "ratings", "avg_rating", "default", "style")
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


@app.get("/api/formats")
def formats_list():
    return F.public_list()


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
    samp = {"temperature": r.temperature, "max_tokens": r.max_tokens}
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
        cmd = [PY, "feedback_train.py", "--run", d] + (["--gate"] if s.get("gate_on_feedback", True) else [])
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
    epochs: float = 3


@app.post("/api/feedback/train")
def feedback_train(r: FeedbackTrainReq):
    d = run_dir(r.model)
    if db.pending_feedback(r.model) == 0:
        err("No new ratings for this model yet.")
    return _launch("feedback", r.model, [PY, "feedback_train.py", "--run", d, "--epochs", str(r.epochs)],
                   os.path.join(d, "log.txt"), run=r.model)


class SettingsReq(BaseModel):
    auto_train: bool | None = None
    auto_threshold: int | None = None
    gate_on_feedback: bool | None = None


@app.post("/api/settings")
def settings(r: SettingsReq):
    return db.set_settings(**{k: v for k, v in r.model_dump().items() if v is not None})


# ---------------------------------------------------------------- corpus
RAW = os.path.join(ROOT, "data", "raw")
EOT = "<|endoftext|>"
SOURCE_NOTES = {
    "gutenberg": "Project Gutenberg books (public domain)", "fineweb": "FineWeb-Edu web text (ODC-By)",
    "custom": "Your own files and links", "news": "News & web crawler", "pg19": "PG-19 books (Apache 2.0)",
    "standard_ebooks": "Standard Ebooks (CC0)", "common_corpus": "Common Corpus, Open Culture (public domain)",
    "archive": "Internet Archive books (public domain)", "cosmopedia": "Cosmopedia (Apache 2.0)",
}


def corpus_dir(source):
    """data/raw/<source> (one folder level, or news/<id>); refuses anything outside data/raw."""
    rel = os.path.normpath(source.strip().replace("\\", "/"))
    d = os.path.normpath(os.path.join(RAW, rel))
    if not rel or rel.startswith("..") or os.path.isabs(rel) or not d.startswith(RAW + os.sep) or not os.path.isdir(d):
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
    threading.Thread(target=crawl_scheduler, daemon=True, name="crawl-scheduler").start()


@app.on_event("shutdown")
def _stop_job():
    """Closing the Studio (Ctrl+C) stops the running job too; it keeps its last checkpoint."""
    if jobs.running():
        print("stopping the running job…", flush=True)
        jobs.stop(grace=60)


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


@app.post("/api/eval")
def eval_start(r: EvalReq):
    d = run_dir(r.model)
    cmd = [PY, "evaluate.py", "--run", d] + (["--quick"] if r.quick else []) + \
          (["--max-tokens", str(r.max_tokens)] if r.max_tokens else [])
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
def make_gguf(name: str):
    d = run_dir(name)
    return _launch("export", name, [PY, "export_gguf.py", "--run", d], os.path.join(d, "log.txt"), run=name)


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
