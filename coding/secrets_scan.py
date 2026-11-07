"""Keep passwords, keys and tokens out of the training data.

Two layers:
  * skip_file(): files that exist to hold secrets (.env, private keys, keystores, cloud credentials)
    are never read at all;
  * scrub(): in every other file, known token formats and `password = "..."`-style assignments are
    replaced with <REDACTED>. A file that contains a private key block is dropped entirely.

This is best effort, not a guarantee: a model can memorise what it trains on, so only add folders
whose code you are happy to have in the model."""
import fnmatch, os, re

SECRET_FILES = [
    ".env", ".env.*", "*.env", ".envrc", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.kdbx",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.ppk", ".npmrc", ".pypirc", ".netrc", "_netrc",
    ".git-credentials", ".dockercfg", "credentials", "credentials.json", "credentials.*", "secrets.*",
    "secret.*", "*secret*.json", "*secrets*.y*ml", "service-account*.json", "*.tfstate", "*.tfstate.*",
    "*.tfvars", "*.ovpn", "htpasswd", ".htpasswd", "wp-config.php", "local.settings.json",
    "appsettings.*.json", "keystore.properties", "google-services.json", "GoogleService-Info.plist",
]
REDACTED = "<REDACTED>"

PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----")
TOKENS = [re.compile(p) for p in (
    r"\bAKIA[0-9A-Z]{16}\b", r"\bASIA[0-9A-Z]{16}\b",                         # AWS access key ids
    r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b", r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b",  # GitHub
    r"\bglpat-[A-Za-z0-9_-]{20,}\b",                                            # GitLab
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b", r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+",
    r"\bAIza[0-9A-Za-z_-]{35}\b",                                              # Google API key
    r"\bya29\.[0-9A-Za-z_-]{20,}\b",                                            # Google OAuth token
    r"\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}\b",                               # Stripe
    r"\bsk-ant-[A-Za-z0-9_-]{20,}\b", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b",  # Anthropic / OpenAI
    r"\bhf_[A-Za-z0-9]{30,}\b",                                                 # Hugging Face
    r"\bnpm_[A-Za-z0-9]{36}\b", r"\bpypi-[A-Za-z0-9_-]{50,}\b",
    r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",                           # SendGrid
    r"\bAC[0-9a-f]{32}\b", r"\bSK[0-9a-f]{32}\b",                              # Twilio
    r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",   # JWT
    r"\b[MN][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,38}\b",                       # Discord bot token
)]
# key = "value" / key: 'value' / KEY=value where the key name says it is secret
ASSIGN = re.compile(
    r"""(?ix)
    (\b[\w.-]*(?:password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?key|private[_-]?key|auth[_-]?token|
        access[_-]?token|refresh[_-]?token|client[_-]?secret|bearer|token|credentials?|conn(?:ection)?[_-]?string)
        [\w.-]*["']?\s*(?::=|=>|=|:)\s*)
    (["'`])([^"'`\n]{6,})\2""")
ENV_ASSIGN = re.compile(
    r"(?im)^(\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*\s*=\s*)"
    r"([^\s\"'#$][^\s#]{5,})")
URL_CRED = re.compile(r"\b([a-z][a-z0-9+.-]*://[^/\s:@]+:)([^@\s/]{3,})(@)", re.I)
PLACEHOLDER = re.compile(r"(?i)^(?:x+|\*+|\.+|<[^>]*>|\{\{.*\}\}|\$\{.*\}|%\(.*\)s|your[_-].*|changeme|example.*|"
                         r"dummy.*|test.*|placeholder|redacted|none|null|true|false|os\.environ.*|process\.env.*)$")


def skip_file(path):
    """True for files that exist to hold secrets."""
    base = os.path.basename(path)
    low = base.lower()
    if low in (".env.example", ".env.sample", ".env.template", ".env.dist"):
        return False
    return any(fnmatch.fnmatch(low, p.lower()) for p in SECRET_FILES)


def scrub(text):
    """(clean text, number of secrets replaced), or (None, 0) if the file holds a private key."""
    if PRIVATE_KEY.search(text):
        return None, 0
    before = text.count(REDACTED)
    for rx in TOKENS:
        text = rx.sub(REDACTED, text)

    def assign(m):
        val = m.group(3).strip()
        if REDACTED in val or PLACEHOLDER.match(val) or " " in val:
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{REDACTED}{m.group(2)}"

    text = ASSIGN.sub(assign, text)
    text = ENV_ASSIGN.sub(lambda m: m.group(0) if PLACEHOLDER.match(m.group(2)) else m.group(1) + REDACTED, text)
    text = URL_CRED.sub(lambda m: m.group(0) if PLACEHOLDER.match(m.group(2)) else m.group(1) + REDACTED + m.group(3),
                        text)
    return text, text.count(REDACTED) - before
