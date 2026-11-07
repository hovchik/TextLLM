"""Accounts, roles and login sessions for the Studio.

Two roles:
  admin  everything: training, data, analytics, feedback, users
  user   only Chat and their own Settings page

Passwords are stored as salted PBKDF2-SHA256 hashes. Sessions are random tokens kept in an
HttpOnly cookie; only their SHA-256 is stored in the database.

Create accounts from the command line:
    python -m studio.auth add alice --admin       # asks for a password
    python -m studio.auth add bob                 # regular user
    python -m studio.auth list
    python -m studio.auth passwd bob
or open the app: if there are no accounts yet, the first screen creates the admin.
"""
import hashlib, hmac, json, secrets, threading, time

from studio import db

ROLES = ("admin", "user")
COOKIE = "studio_session"
SESSION_DAYS = 30
ITERATIONS = 240_000
DEFAULT_PREFS = {"temperature": 0.8, "max_tokens": 300, "model": ""}

db.ex_script("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT,
    pw_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    prefs TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    last_login REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created REAL NOT NULL,
    expires REAL NOT NULL
);
""")


# ---------------------------------------------------------------- passwords
def hash_password(pw):
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${h}"


def check_password(pw, stored):
    try:
        _, it, salt, h = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it)).hex()
        return hmac.compare_digest(got, h)
    except (ValueError, AttributeError):
        return False


def validate(username, password):
    username = (username or "").strip()
    if not 2 <= len(username) <= 40 or not all(c.isalnum() or c in "-_.@" for c in username):
        raise ValueError("Username: 2–40 letters, digits or - _ . @")
    if len(password or "") < 6:
        raise ValueError("Password must be at least 6 characters.")
    return username


# ---------------------------------------------------------------- users
def public(u):
    """User row without the password hash, with prefs decoded."""
    if not u:
        return None
    prefs = {**DEFAULT_PREFS, **json.loads(u.get("prefs") or "{}")}
    return {"id": u["id"], "username": u["username"], "display_name": u["display_name"] or u["username"],
            "role": u["role"], "prefs": prefs, "created": u["created"], "last_login": u["last_login"]}


def count_users():
    return db.q("SELECT COUNT(*) n FROM users", one=True)["n"]


def get_user(uid):
    return db.q("SELECT * FROM users WHERE id=?", (uid,), one=True)


def list_users():
    rows = db.q("SELECT u.*, (SELECT COUNT(*) FROM conversations c WHERE c.user_id=u.id) chats, "
                "(SELECT COUNT(m.rating) FROM messages m JOIN conversations c ON c.id=m.conversation_id "
                " WHERE c.user_id=u.id) ratings FROM users u ORDER BY u.id")
    return [{**public(r), "chats": r["chats"], "ratings": r["ratings"]} for r in rows]


class NotFirst(Exception):
    """create_user(only_if_first=True) found that accounts already exist."""


def create_user(username, password, role="user", display_name=None, only_if_first=False):
    username = validate(username, password)
    if role not in ROLES:
        raise ValueError("Role must be admin or user.")
    pw_hash = hash_password(password)  # slow: do it before taking the lock
    # Check and insert in one locked transaction, so two first-run setups can't both create an admin.
    with db._lock, db.conn() as c:
        first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        if only_if_first and not first:
            raise NotFirst()
        if c.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError(f"The username {username} is taken.")
        uid = c.execute("INSERT INTO users(username, display_name, pw_hash, role, created) VALUES(?,?,?,?,?)",
                        (username, (display_name or "").strip() or None, pw_hash, role, time.time())).lastrowid
        if first and role == "admin":  # chats from before accounts existed belong to the first admin
            c.execute("UPDATE conversations SET user_id=? WHERE user_id IS NULL", (uid,))
    return public(get_user(uid))


def admins():
    return db.q("SELECT COUNT(*) n FROM users WHERE role='admin'", one=True)["n"]


def update_user(uid, role=None, password=None, display_name=None, prefs=None):
    u = get_user(uid)
    if not u:
        raise LookupError("No such user.")
    if role is not None:
        if role not in ROLES:
            raise ValueError("Role must be admin or user.")
        if u["role"] == "admin" and role != "admin" and admins() <= 1:
            raise ValueError("Keep at least one admin.")
        db.ex("UPDATE users SET role=? WHERE id=?", (role, uid))
    if password is not None:
        validate(u["username"], password)
        db.ex("UPDATE users SET pw_hash=? WHERE id=?", (hash_password(password), uid))
        db.ex("DELETE FROM sessions WHERE user_id=?", (uid,))  # sign out everywhere
    if display_name is not None:
        db.ex("UPDATE users SET display_name=? WHERE id=?", (display_name.strip()[:60] or None, uid))
    if prefs is not None:
        cur = json.loads(u["prefs"] or "{}")
        cur.update({k: v for k, v in prefs.items() if k in DEFAULT_PREFS})
        db.ex("UPDATE users SET prefs=? WHERE id=?", (json.dumps(cur), uid))
    return public(get_user(uid))


def delete_user(uid):
    u = get_user(uid)
    if not u:
        raise LookupError("No such user.")
    if u["role"] == "admin" and admins() <= 1:
        raise ValueError("Keep at least one admin.")
    db.ex("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id=?)", (uid,))
    db.ex("DELETE FROM conversations WHERE user_id=?", (uid,))
    db.ex("DELETE FROM sessions WHERE user_id=?", (uid,))
    db.ex("DELETE FROM users WHERE id=?", (uid,))


# ---------------------------------------------------------------- sessions
def _th(token):
    return hashlib.sha256(token.encode()).hexdigest()


# Wrong passwords per (username, client address): after MAX_FAILS in a row the pair is locked for
# LOCK_SECONDS. Kept in memory: a restart clears it, which is fine for a brute-force brake.
MAX_FAILS, LOCK_SECONDS = 8, 60
_fails = {}
_fails_lock = threading.Lock()


def _key(username, client):
    return ((username or "").strip().lower(), client or "")


def locked_out(username, client=""):
    k = _key(username, client)
    with _fails_lock:
        n, last = _fails.get(k, (0, 0))
        if n >= MAX_FAILS and time.time() - last < LOCK_SECONDS:
            return True
        if n >= MAX_FAILS:
            _fails.pop(k, None)  # the lock has run out
        return False


def _failed(username, client):
    k, now = _key(username, client), time.time()
    with _fails_lock:
        _fails[k] = (_fails.get(k, (0, 0))[0] + 1, now)
        if len(_fails) > 10000:  # keep memory bounded if someone sprays usernames
            for old in [kk for kk, (_, t) in _fails.items() if now - t > LOCK_SECONDS]:
                _fails.pop(old, None)


def login(username, password, client=""):
    u = db.q("SELECT * FROM users WHERE username=?", ((username or "").strip(),), one=True)
    if not u or not check_password(password or "", u["pw_hash"]):
        _failed(username, client)
        time.sleep(0.5)  # slow down guessing
        return None, None
    with _fails_lock:
        _fails.pop(_key(username, client), None)
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.ex("DELETE FROM sessions WHERE expires<?", (now,))
    db.ex("INSERT INTO sessions(token_hash, user_id, created, expires) VALUES(?,?,?,?)",
          (_th(token), u["id"], now, now + SESSION_DAYS * 86400))
    db.ex("UPDATE users SET last_login=? WHERE id=?", (now, u["id"]))
    return token, public(u)


def user_for_token(token):
    if not token:
        return None
    u = db.q("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>?",
             (_th(token), time.time()), one=True)
    return public(u)


def logout(token):
    if token:
        db.ex("DELETE FROM sessions WHERE token_hash=?", (_th(token),))


# ---------------------------------------------------------------- CLI
def _cli():
    import argparse, getpass

    ap = argparse.ArgumentParser(prog="python -m studio.auth", description="Manage Studio accounts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="create an account")
    a.add_argument("username")
    a.add_argument("--admin", action="store_true", help="give the account the admin role")
    a.add_argument("--password", help="(default: ask)")
    p = sub.add_parser("passwd", help="set a new password")
    p.add_argument("username")
    p.add_argument("--password")
    r = sub.add_parser("role", help="change an account's role")
    r.add_argument("username")
    r.add_argument("role", choices=ROLES)
    d = sub.add_parser("delete", help="delete an account and its chats")
    d.add_argument("username")
    sub.add_parser("list", help="list accounts")
    args = ap.parse_args()

    def ask():
        pw = getpass.getpass("Password: ")
        if pw != getpass.getpass("Repeat: "):
            raise SystemExit("Passwords don't match.")
        return pw

    def find(name):
        u = db.q("SELECT * FROM users WHERE username=?", (name,), one=True)
        if not u:
            raise SystemExit(f"No account named {name}.")
        return u

    try:
        if args.cmd == "add":
            u = create_user(args.username, args.password or ask(), "admin" if args.admin else "user")
            print(f"Created {u['role']} {u['username']}.")
        elif args.cmd == "passwd":
            update_user(find(args.username)["id"], password=args.password or ask())
            print("Password changed.")
        elif args.cmd == "role":
            update_user(find(args.username)["id"], role=args.role)
            print(f"{args.username} is now {args.role}.")
        elif args.cmd == "delete":
            delete_user(find(args.username)["id"])
            print("Deleted.")
        else:
            for u in list_users():
                print(f"{u['id']:>3}  {u['username']:<24} {u['role']:<6} {u['chats']} chats")
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    _cli()
