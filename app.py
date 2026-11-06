"""Text LLM Studio: web app for data, training, analytics, chat and feedback.

    pip install -r requirements.txt
    python app.py                 # then open http://127.0.0.1:7860
    python app.py --port 8000 --host 0.0.0.0

Sign-in is required. The first visit creates the admin account; admins add more people on the
Users page (or with `python -m studio.auth add NAME`). Regular users only see Chat and Settings.
"""
import argparse, json, os, re, shutil, threading, time

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat_engine import ROOT, RUNS, Engine, list_models
from studio import auth, db, feedback, jobs

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
    ("POST", r"/api/chat"), ("POST", r"/api/messages/\d+/rate"),
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
    return {"job": jobs.status(), "data": jobs.data_status(), "settings": s,
            "pending_feedback": db.pending_feedback(), "user": me(request)}


class DataReq(BaseModel):
    step: str            # download | tokenizer | encode
    books: int = 3000
    web_gb: float = 0.3
    vocab: int = 8192


@app.post("/api/data")
def data(r: DataReq):
    cmds = {
        "download": [PY, "data/build_corpus.py", "--gutenberg-max", str(r.books), "--fineweb-gb", str(r.web_gb)],
        "tokenizer": [PY, "tokenizer/train_tokenizer.py", "--vocab", str(r.vocab)],
        "encode": [PY, "tokenizer/train_tokenizer.py", "--encode"],
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
    base: str = "Qwen/Qwen2.5-0.5B-Instruct"
    examples: int = 4000
    seq: int = 512


@app.post("/api/train")
def train(r: TrainReq):
    name = "".join(c if c.isalnum() or c in "-_." else "-" for c in r.name.strip()) or "my-model"
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    if r.mode == "scratch":
        if not os.path.exists(os.path.join(ROOT, "data/bin/train.bin")):
            err("Prepare the training data first (Training → Data, steps 1–3).")
        json.dump({"type": "scratch", "tok": os.path.join(ROOT, "tokenizer/bpe.json"), "preset": r.preset},
                  open(os.path.join(out, "meta.json"), "w"))
        cmd = [PY, "train.py", "--out", out, "--preset", r.preset, "--vocab", str(r.vocab),
               "--max-steps", str(r.steps), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--warmup", str(max(10, r.steps // 40)), "--eval-every", str(max(10, r.steps // 80))]
        if r.resume:
            cmd.append("--resume")
    else:
        import glob
        if not glob.glob(os.path.join(ROOT, "data/raw/**/*.txt"), recursive=True):
            err("Add some text first: download books (Training → Data) or add your own texts.")
        cmd = [PY, "finetune/finetune_lora.py", "--out", out, "--base", r.base, "--examples", str(r.examples),
               "--seq", str(r.seq), "--batch", str(r.batch), "--grad-accum", str(r.accum), "--lr", str(r.lr),
               "--max-steps", str(r.steps), "--eval-every", str(max(5, r.steps // 20)),
               "--save-every", str(max(5, r.steps // 10))]
    mp = os.path.join(out, "metrics.jsonl")
    if not r.resume and os.path.exists(mp):
        os.rename(mp, os.path.join(out, f"metrics.{int(time.time())}.jsonl"))
    return _launch("training", name, cmd, os.path.join(out, "log.txt"), run=name)


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
    w = os.path.join(d, "ckpt.pt") if os.path.exists(os.path.join(d, "ckpt.pt")) else os.path.join(d, "adapter")
    # adapter dirs: overwriting a file inside doesn't change the dir's mtime, so look at the files
    files = [os.path.join(w, f) for f in os.listdir(w)] if os.path.isdir(w) else [w]
    stamp = max(os.path.getmtime(f) for f in files)
    with _engine_lock:
        cached = ENGINES.get(name)
        if not cached or cached[0] != stamp:  # reload after (feedback) training
            ENGINES.clear()
            ENGINES[name] = (stamp, Engine(d))
        return ENGINES[name][1]


@app.get("/api/models")
def models(request: Request):
    out = []
    for label, d in list_models().items():
        s = jobs.run_summary(d)
        stats = db.q("SELECT COUNT(rating) n, AVG(rating) avg FROM messages WHERE model=? AND rating IS NOT NULL",
                     (s["name"],), one=True)
        out.append({**s, "label": label, "ratings": stats["n"], "avg_rating": stats["avg"],
                    "pending_feedback": db.pending_feedback(s["name"])})
    if not is_admin(request):  # chat only needs to pick a model
        keep = ("name", "type", "label", "level", "ratings", "avg_rating")
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
    temperature: float = 0.8
    max_tokens: int = 300


@app.post("/api/chat")
def chat(r: ChatReq, request: Request):
    eng = engine(r.model)
    if r.regenerate:
        um = db.q("SELECT * FROM messages WHERE id=? AND role='user'", (r.regenerate,), one=True) or err("not found", 404)
        own_conversation(request, um["conversation_id"])
        cid, uid, text = um["conversation_id"], um["id"], um["content"]
    else:
        text = r.message.strip() or err("Empty message")
        if r.conversation_id:
            own_conversation(request, r.conversation_id)
        cid = r.conversation_id or db.new_conversation(text, r.model, me(request)["id"])
        uid = db.add_message(cid, "user", text)
    hist = db.history(cid, before_id=uid)

    def gen():
        yield json.dumps({"type": "start", "conversation_id": cid, "user_id": uid}) + "\n"
        sent, full = "", ""
        try:
            for full in eng.stream(hist, text, r.max_tokens, r.temperature):
                if eng.meta["type"] == "scratch" and full.startswith(text):
                    full = full[len(text):]
                if len(full) > len(sent):
                    yield json.dumps({"type": "delta", "text": full[len(sent):]}) + "\n"
                    sent = full
        except Exception as e:  # report instead of breaking the stream
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        finally:  # also runs if the browser stops the stream: keep what was written
            mid = db.add_message(cid, "assistant", full.strip() or "…", model=r.model, prompt_id=uid)
        yield json.dumps({"type": "done", "message_id": mid}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


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
        jobs.launch("feedback", model, [PY, "feedback_train.py", "--run", d], os.path.join(d, "log.txt"), run=model)
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


@app.post("/api/settings")
def settings(r: SettingsReq):
    return db.set_settings(**{k: v for k, v in r.model_dump().items() if v is not None})


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
        "trend": trend, "val_curves": curves, "jobs": jobs.history(30),
        "train_hours": sum(r["train_seconds"] for r in rs) / 3600,
        "pending_feedback": db.pending_feedback(),
    }


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
