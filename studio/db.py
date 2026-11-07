"""SQLite storage for conversations, messages, ratings and settings."""
import json, os, sqlite3, threading, time

from chat_engine import ROOT

DB_PATH = os.path.join(ROOT, "runs", "studio.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    model TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                 -- user | assistant
    content TEXT NOT NULL,
    model TEXT,                         -- run name that produced an assistant message
    prompt_id INTEGER,                  -- assistant: the user message it answers
    created REAL NOT NULL,
    rating INTEGER,                     -- 1..5 stars, NULL = not rated
    comment TEXT,
    correction TEXT,                    -- the user's better answer
    rated_at REAL,
    trained_in TEXT                     -- run/time it was used for training
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS pairs (          -- blind A/B comparisons (evaluate.py --pairs)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_key TEXT, format TEXT, prompt TEXT,
    run_a TEXT NOT NULL, run_b TEXT NOT NULL,   -- run_a = the candidate, run_b = the current model
    text_a TEXT, text_b TEXT,
    vote TEXT,                                  -- a | b | tie | NULL (not voted yet)
    voter TEXT, created REAL NOT NULL, voted_at REAL
);
"""

DEFAULT_SETTINGS = {"device": "auto",  # where training and chat run: auto | cpu | cuda | cuda:1 | mps | xpu
                    "auto_train": True, "auto_threshold": 10, "default_model": None, "gate_on_feedback": True,
                    "feedback_epochs": 3, "feedback_lr": 5e-5, "feedback_beta": 0.1,  # feedback_train.py
                    # news & web crawler (data/crawl.py), run by the scheduler in app.py
                    "crawl_enabled": False, "crawl_every_hours": 6, "crawl_learn_model": "",
                    "crawl_min_docs": 20, "crawl_passes": 2, "crawl_last_start": 0,
                    "crawl_languages": "", "crawl_gate": True, "crawl_max_drop": 5}


_conn = {}  # DB_PATH -> the process's one connection (every use holds _lock)


def conn():
    """The shared connection. Opening one per query cost more than most queries (the sign-in
    check runs one on every request). `with conn() as c` commits, or rolls back on an error."""
    c = _conn.get(DB_PATH)
    if c is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        _conn[DB_PATH] = c
    return c


def init():
    """Create the tables and add columns that newer versions use (run at import)."""
    with _lock, conn() as c:
        c.executescript(SCHEMA)
        if "user_id" not in [r["name"] for r in c.execute("PRAGMA table_info(conversations)")]:
            c.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")  # owner (studio/auth.py)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(messages)")]
        for col, typ in (("format", "TEXT"), ("fields", "TEXT"), ("check_score", "REAL")):
            if col not in cols:  # writing format of a request/answer, its form fields (JSON), automatic score
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {typ}")


init()


def q(sql, args=(), one=False):
    with _lock, conn() as c:
        cur = c.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    with _lock, conn() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid


def ex_script(sql):
    with _lock, conn() as c:
        c.executescript(sql)


def get_settings():
    s = dict(DEFAULT_SETTINGS)
    for r in q("SELECT key, value FROM settings"):
        s[r["key"]] = json.loads(r["value"])
    return s


def set_settings(**kw):
    for k, v in kw.items():
        ex("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
           (k, json.dumps(v)))
    return get_settings()


# ---------------------------------------------------------------- conversations
def new_conversation(title, model, user_id=None):
    now = time.time()
    return ex("INSERT INTO conversations(title, model, created, updated, user_id) VALUES(?,?,?,?,?)",
              (title[:80], model, now, now, user_id))


def add_message(conv_id, role, content, model=None, prompt_id=None, fmt=None, fields=None, check_score=None):
    ex("UPDATE conversations SET updated=?, model=COALESCE(?, model) WHERE id=?", (time.time(), model, conv_id))
    return ex("INSERT INTO messages(conversation_id, role, content, model, prompt_id, created, format, fields, "
              "check_score) VALUES(?,?,?,?,?,?,?,?,?)",
              (conv_id, role, content, model, prompt_id, time.time(), fmt,
               json.dumps(fields) if fields else None, check_score))


def history(conv_id, before_id=None):
    """Messages for the model's context: each user turn plus its latest/best answer."""
    rows = q("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))
    out, answers = [], {}
    for r in rows:
        if before_id is not None and r["id"] >= before_id:
            break
        if r["role"] == "assistant":
            prev = answers.get(r["prompt_id"])
            if prev is None or (r["rating"] or 3) >= (prev["rating"] or 3):
                answers[r["prompt_id"]] = r
    for r in rows:
        if before_id is not None and r["id"] >= before_id:
            break
        if r["role"] == "user":
            out.append({"role": "user", "content": r["content"]})
            a = answers.get(r["id"])
            if a:
                out.append({"role": "assistant", "content": a["correction"] or a["content"]})
    return out


def rate(msg_id, rating=None, comment=None, correction=None):
    ex("UPDATE messages SET rating=COALESCE(?, rating), comment=COALESCE(?, comment), "
       "correction=COALESCE(?, correction), rated_at=?, trained_in=NULL "  # new feedback → learn it again
       "WHERE id=? AND role='assistant'",
       (rating, comment, correction or None, time.time(), msg_id))
    return q("SELECT * FROM messages WHERE id=? AND role='assistant'", (msg_id,), one=True)


def pending_feedback(model=None):
    sql = ("SELECT COUNT(*) n FROM messages WHERE role='assistant' AND trained_in IS NULL "
           "AND (rating IS NOT NULL OR correction IS NOT NULL)")
    args = ()
    if model:
        sql += " AND model=?"
        args = (model,)
    return q(sql, args, one=True)["n"]


# ---------------------------------------------------------------- blind A/B pairs
def add_pair(prompt_key, fmt, prompt, run_a, run_b, text_a, text_b):
    return ex("INSERT INTO pairs(prompt_key, format, prompt, run_a, run_b, text_a, text_b, created) "
              "VALUES(?,?,?,?,?,?,?,?)", (prompt_key, fmt, prompt, run_a, run_b, text_a, text_b, time.time()))


def vote_pair(pid, vote, voter):
    if vote not in ("a", "b", "tie"):
        raise ValueError("vote must be a, b or tie")
    ex("UPDATE pairs SET vote=?, voter=?, voted_at=? WHERE id=?", (vote, voter, time.time(), pid))


def next_pair():
    return q("SELECT * FROM pairs WHERE vote IS NULL ORDER BY RANDOM() LIMIT 1", one=True)


def pair_stats(run_a=None, run_b=None, since=0):
    """Wins of run_a (candidate) over run_b; ties count half. since = ignore pairs made before."""
    sql = ("SELECT run_a, run_b, format, SUM(vote='a') a, SUM(vote='b') b, SUM(vote='tie') t, "
           "SUM(vote IS NULL) open FROM pairs WHERE created >= ?")
    args = (since,)
    if run_a and run_b:
        sql += " AND run_a=? AND run_b=?"
        args = (since, run_a, run_b)
    rows = q(sql + " GROUP BY run_a, run_b, format", args)
    out = {}
    for r in rows:
        k = (r["run_a"], r["run_b"])
        o = out.setdefault(k, {"run_a": r["run_a"], "run_b": r["run_b"], "a": 0, "b": 0, "tie": 0, "open": 0,
                               "per_format": {}})
        for x, y in (("a", "a"), ("b", "b"), ("t", "tie"), ("open", "open")):
            o[y] += r[x] or 0
        n = (r["a"] or 0) + (r["b"] or 0) + (r["t"] or 0)
        o["per_format"][r["format"]] = round(((r["a"] or 0) + 0.5 * (r["t"] or 0)) / n, 3) if n else None
    for o in out.values():
        n = o["a"] + o["b"] + o["tie"]
        o["votes"] = n
        o["win_rate"] = round((o["a"] + 0.5 * o["tie"]) / n, 3) if n else None
    return list(out.values())
