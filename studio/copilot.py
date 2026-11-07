"""Studio API for the copilot (code) model: code sources, the folder picker and data preparation.

Routes (admin only, like every route not listed in app.USER_ROUTES):
  GET    /api/code                 sources on disk, saved settings, tokenizer/encode status
  POST   /api/code/config          save the settings (local folders, GitHub, Hugging Face, filters)
  POST   /api/code/build           save the settings and run data/build_code.py for some sources
  POST   /api/code/prepare         from-scratch copilots: code tokenizer or encoding (with FIM)
  DELETE /api/code/sources/{id}    delete one collected source (local/<name>, github, hf)
  GET    /api/fs/dirs?path=        list the folders on this PC, for picking project folders
  POST   /api/code/auto            automatic updates: save the settings
  POST   /api/code/auto/run        automatic updates: one run now
  POST   /api/code/auto/schedule   automatic updates: register with / remove from the OS scheduler
"""
import glob, json, os, re, shutil, string, sys, time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chat_engine import ROOT, RUNS
from core import paths as core_paths
from data import code_update as auto
from data.build_code import parse_topics
from studio import jobs

router = APIRouter()
CODE = os.path.join(ROOT, "data", "code")
CFG = os.path.join(CODE, "sources.json")
CODE_BIN = os.path.join(ROOT, "data", "bin-code")
CODE_TOK = os.path.join(ROOT, "tokenizer", "bpe-code.json")
LOG = os.path.join(RUNS, "_data", "log.txt")
PERMISSIVE = ["mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "0bsd", "isc", "unlicense",
              "cc0-1.0", "zlib", "bsl-1.0"]


def err(msg, code=400):
    raise HTTPException(code, msg)


def launch(name, cmd, env=None):
    try:
        return jobs.launch("data", name, cmd, LOG, env=env)
    except RuntimeError as e:
        err(str(e), 409)


class CodeCfg(BaseModel):
    local_dirs: list[str] = []
    github_repos: int = 0             # repositories to have in total, found by search
    github_list: str = ""             # owner/name, one per line or comma-separated
    min_stars: int = 50
    max_repo_mb: float = 40
    github_token: str | None = None   # None = keep the saved one, "" = forget it
    hf_gb: float = 0
    hf_mode: str = "stream"
    langs: str = "python,javascript,typescript"
    licenses: list[str] = PERMISSIVE
    max_file_kb: int = 256
    retries: int = 30


def load_cfg():
    try:
        return {**CodeCfg().model_dump(), **json.load(open(CFG))}
    except (OSError, ValueError):
        return CodeCfg().model_dump()


def save_cfg(r: CodeCfg):
    old = load_cfg()
    c = r.model_dump()
    c["github_token"] = (old.get("github_token") or "") if r.github_token is None else r.github_token.strip()
    c["local_dirs"] = list(dict.fromkeys(d.strip().strip('"') for d in r.local_dirs if d.strip()))
    c["licenses"] = [x for x in (x.lower() for x in r.licenses) if x in PERMISSIVE] or PERMISSIVE
    c["langs"] = re.sub(r"[^\w#+, -]", "", r.langs)[:300]
    os.makedirs(CODE, exist_ok=True)
    tmp = CFG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, indent=1)
    os.replace(tmp, CFG)
    return c


def public_cfg(c):
    return {**c, "github_token": None, "github_token_set": bool(c.get("github_token"))}


def _size(d):
    files = glob.glob(os.path.join(d, "**", "shard_*.txt"), recursive=True)
    return len(files), sum(os.path.getsize(f) for f in files)


def _count_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def sources():
    out = []
    for d in sorted(glob.glob(os.path.join(CODE, "local", "*"))):
        info = {}
        try:
            info = json.load(open(os.path.join(d, "info.json")))
        except (OSError, ValueError):
            pass
        _, size = _size(d)
        out.append({"id": f"local/{os.path.basename(d)}", "kind": "local", "label": info.get("name") or os.path.basename(d),
                    "folder": info.get("folder"), "files": info.get("files", 0), "mb": round(size / 1e6, 2),
                    "langs": info.get("langs", {}), "license": info.get("license"), "redacted": info.get("redacted", 0),
                    "skipped": info.get("skipped", {}), "updated": info.get("updated")})
    gh = os.path.join(CODE, "github")
    if os.path.isdir(gh):
        st = {}
        try:
            st = json.load(open(os.path.join(gh, "progress.json")))
        except (OSError, ValueError):
            pass
        rows, lic = {}, {}
        p = os.path.join(gh, "attribution.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    rows[r["repo"]] = r["license"]  # a refreshed repository has several rows; the last one counts
        for v in rows.values():
            lic[v] = lic.get(v, 0) + 1
        _, size = _size(gh)
        out.append({"id": "github", "kind": "github", "label": "Public GitHub (permissive licences)",
                    "repos": len(st.get("done", [])), "rejected": len(st.get("rejected", {})),
                    "files": st.get("files", 0), "mb": round(size / 1e6, 2), "licenses": lic})
    hf = os.path.join(CODE, "hf")
    if os.path.isdir(hf):
        n, size = _size(hf)
        st = {}
        try:
            st = json.load(open(os.path.join(hf, "progress.json")))
        except (OSError, ValueError):
            pass
        out.append({"id": "hf", "kind": "hf", "label": "GitHub code from Hugging Face (permissive rows)",
                    "files": st.get("docs", 0), "mb": round(size / 1e6, 2)})
    return out


def prepared():
    tb = os.path.join(CODE_BIN, "train.bin")
    return {"tokenizer": os.path.exists(CODE_TOK), "vocab": jobs._vocab_size(CODE_TOK) if os.path.exists(CODE_TOK) else None,
            "train_tokens": os.path.getsize(tb) // 2 if os.path.exists(tb) else 0}


def has_code():
    return bool(glob.glob(os.path.join(CODE, "**", "shard_*.txt"), recursive=True))


@router.get("/api/code")
def code_state():
    return {"config": public_cfg(load_cfg()), "sources": sources(), "prepared": prepared(),
            "licenses": PERMISSIVE, "attribution": os.path.exists(os.path.join(CODE, "ATTRIBUTION.md")),
            "auto": auto_state()}


@router.post("/api/code/config")
def code_config(r: CodeCfg):
    return {"config": public_cfg(save_cfg(r))}


class BuildReq(CodeCfg):
    what: str = "all"   # all | local | github | hf | one local folder's path


@router.post("/api/code/build")
def code_build(r: BuildReq):
    c = save_cfg(r)
    cmd = [jobs.PY, "data/build_code.py", "--langs", c["langs"], "--licenses", ",".join(c["licenses"]),
           "--max-file-kb", str(max(8, min(4096, c["max_file_kb"]))), "--retries", str(max(0, min(1000, c["retries"])))]
    what = r.what
    run_local = what in ("all", "local") or what in c["local_dirs"]
    run_gh = what in ("all", "github") and (c["github_repos"] > 0 or c["github_list"].strip())
    run_hf = what in ("all", "hf") and c["hf_gb"] > 0
    if run_local:
        dirs = c["local_dirs"] if what in ("all", "local") else [what]
        for d in dirs:
            if not os.path.isdir(os.path.expanduser(d)):
                err(f"{d} is not a folder on this PC.")
            cmd += ["--local", d]
        if not dirs and what == "local":
            err("Add a project folder first.")
    if run_gh:
        if c["github_repos"] > 0:
            cmd += ["--github-repos", str(min(100000, c["github_repos"])), "--min-stars", str(max(0, c["min_stars"]))]
        for full in re.split(r"[\s,]+", c["github_list"]):
            full = full.strip().removeprefix("https://github.com/").strip("/")
            if full:
                if not re.fullmatch(r"[\w.-]+/[\w.-]+", full):
                    err(f"{full}: write GitHub repositories as owner/name")
                cmd += ["--github-repo", full]
        cmd += ["--max-repo-mb", str(max(1.0, min(2000.0, c["max_repo_mb"])))]
    if run_hf:
        cmd += ["--hf-gb", str(min(500.0, c["hf_gb"])), "--hf-mode", "download" if c["hf_mode"] == "download" else "stream"]
    if not any(x in cmd for x in ("--local", "--github-repos", "--github-repo", "--hf-gb")):
        err("Nothing to collect: add a project folder, a number of GitHub repositories or repository names, "
            "or Hugging Face GB.")
    env = {"GITHUB_TOKEN": c["github_token"]} if c.get("github_token") else None  # not written to the log
    return launch("code", cmd, env=env)


class PrepareReq(BaseModel):
    step: str                 # tokenizer | encode
    vocab: int = 16384
    sample_files: int = 40
    fim_rate: float = 0.5
    with_text: bool = False   # also learn English from data/raw (comments, docs)
    val_frac: float = 0.005


@router.post("/api/code/prepare")
def code_prepare(r: PrepareReq):
    if not has_code():
        err("Collect some code first (step 1).")
    raw = "data/code" + (",data/raw" if r.with_text and os.path.isdir(os.path.join(ROOT, "data", "raw")) else "")
    if r.step == "tokenizer":
        cmd = [jobs.PY, "tokenizer/train_tokenizer.py", "--code", "--raw", raw, "--vocab", str(max(1000, min(65000, r.vocab))),
               "--out", "tokenizer/bpe-code.json", "--sample-files", str(max(1, r.sample_files))]
    elif r.step == "encode":
        if not os.path.exists(CODE_TOK):
            err("Build the code tokenizer first.")
        cmd = [jobs.PY, "tokenizer/train_tokenizer.py", "--encode", "--raw", raw, "--tok", "tokenizer/bpe-code.json",
               "--bin", "data/bin-code", "--fim-rate", str(max(0.0, min(1.0, r.fim_rate))),
               "--val-frac", str(max(0.0001, min(0.5, r.val_frac)))]
    else:
        err("unknown step")
    return launch(f"code {r.step}", cmd)


@router.delete("/api/code/sources/{sid:path}")
def code_delete(sid: str):
    parts = sid.strip().replace("\\", "/").split("/")
    d = core_paths.under(CODE, *parts)
    ok = parts in (["github"], ["hf"]) or (len(parts) == 2 and parts[0] == "local")
    if not ok or d is None or not os.path.isdir(d):
        err(f"no code source named {sid}", 404)
    if jobs.running() and jobs.JOB["kind"] == "data":
        err("Stop the running data job first.", 409)
    shutil.rmtree(d)
    if parts == ["github"] and os.path.exists(os.path.join(CODE, "ATTRIBUTION.md")):
        os.remove(os.path.join(CODE, "ATTRIBUTION.md"))
    return code_state()


@router.get("/api/fs/dirs")
def fs_dirs(path: str = ""):
    """Folders inside `path` (the home folder when empty; drive letters on Windows at the top)."""
    if not path.strip():
        path = os.path.expanduser("~")
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
    if not os.path.isdir(path):
        err(f"{path} is not a folder", 404)
    try:
        names = sorted((n for n in os.listdir(path) if not n.startswith(".") and os.path.isdir(os.path.join(path, n))),
                       key=str.lower)
    except OSError as e:
        err(f"can't open {path}: {e.strerror}", 403)
    parent = os.path.dirname(path)
    drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")] if sys.platform == "win32" else []
    marks = ("package.json", "pyproject.toml", "setup.py", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml",
             "build.gradle", ".git", "CMakeLists.txt", "Makefile", "composer.json", "Gemfile")
    return {"path": path, "parent": parent if parent != path else None, "drives": drives,
            "home": os.path.expanduser("~"),
            "project": any(os.path.exists(os.path.join(path, m)) for m in marks),
            "dirs": [{"name": n, "path": os.path.join(path, n)} for n in names[:500]]}


# ---------------------------------------------------------------- automatic updates (data/code_update.py)
class AutoCfg(BaseModel):
    enabled: bool | None = None
    every_hours: float | None = None
    github_new: int | None = None
    refresh: bool | None = None
    refresh_limit: int | None = None
    topics: str | None = None
    local: bool | None = None
    train_model: str | None = None   # "" = only collect
    train_steps: int | None = None
    min_files: int | None = None
    mix: float | None = None


def auto_next(s):
    """When the Studio's own scheduler runs the updates next; None when off or when the OS scheduler does it."""
    if not s["enabled"] or auto.schedule_status()["installed"]:
        return None
    return auto.load_state()["last_start"] + float(s["every_hours"]) * 3600


def copilot_models():
    return [r["name"] for r in jobs.runs() if r.get("task") == "code"]


def auto_state():
    s = auto.load_settings()
    return {"settings": s, "state": auto.load_state(), "next": auto_next(s), "schedule": auto.schedule_status(),
            "models": copilot_models(), "log": jobs.tail(auto.LOG, 150)}


@router.post("/api/code/auto")
def auto_config(r: AutoCfg):
    s = {**auto.load_settings(), **r.model_dump(exclude_none=True)}
    s["every_hours"] = max(1.0, min(24 * 7, float(s["every_hours"])))
    s["github_new"] = max(0, min(1000, int(s["github_new"])))
    s["refresh_limit"] = max(0, min(100000, int(s["refresh_limit"])))
    s["topics"] = ",".join(parse_topics(s["topics"]))[:500]
    s["train_steps"] = max(1, min(100000, int(s["train_steps"])))
    s["min_files"] = max(1, int(s["min_files"]))
    s["mix"] = max(0.0, min(10.0, float(s["mix"])))
    if s["train_model"]:
        if s["train_model"] not in copilot_models():
            err(f"{s['train_model']} is not a copilot model (train one in Training → Copilot first).")
    auto.save_settings(s)
    return auto_state()


def start_auto():
    s = auto.load_settings()
    run = s["train_model"] if s["train_model"] in copilot_models() else None
    try:
        st = jobs.launch("data", "code updates", [jobs.PY, "data/code_update.py"], auto.LOG, run=run)
    except RuntimeError as e:
        err(str(e), 409)
    auto.update_state(last_start=time.time())  # the in-app scheduler waits a full interval again
    return st


@router.post("/api/code/auto/run")
def auto_run():
    return start_auto()


@router.post("/api/code/auto/schedule")
def auto_schedule(install: bool = True):
    s = auto.load_settings()
    try:
        msg = auto.install_schedule(s["every_hours"]) if install else auto.uninstall_schedule()
    except FileNotFoundError:
        err("No scheduler found on this computer (schtasks on Windows, crontab on Linux/macOS).")
    except RuntimeError as e:
        err(f"Could not change the schedule: {e}")
    return {"message": msg, **auto_state()}


def auto_scheduler():
    """Started by app.py: runs the code updates when they are due (waits while another job runs)."""
    while True:
        try:
            nxt = auto_next(auto.load_settings())
            if nxt is not None and time.time() >= nxt and not jobs.running() and not jobs.external_job():
                start_auto()
        except Exception as e:  # keep the scheduler alive whatever happens
            print("code updates scheduler:", e, flush=True)
        time.sleep(60)
