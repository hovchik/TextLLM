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
"""

DEFAULT_SETTINGS = {"auto_train": True, "auto_threshold": 10}


def conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


with conn() as _c:
    _c.executescript(SCHEMA)


def q(sql, args=(), one=False):
    with _lock, conn() as c:
        cur = c.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    with _lock, conn() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid


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
def new_conversation(title, model):
    now = time.time()
    return ex("INSERT INTO conversations(title, model, created, updated) VALUES(?,?,?,?)",
              (title[:80], model, now, now))


def add_message(conv_id, role, content, model=None, prompt_id=None):
    ex("UPDATE conversations SET updated=?, model=COALESCE(?, model) WHERE id=?", (time.time(), model, conv_id))
    return ex("INSERT INTO messages(conversation_id, role, content, model, prompt_id, created) VALUES(?,?,?,?,?,?)",
              (conv_id, role, content, model, prompt_id, time.time()))


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
       "correction=COALESCE(?, correction), rated_at=? WHERE id=? AND role='assistant'",
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
