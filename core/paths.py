"""Keep user-supplied names and paths inside the folder they belong to."""
import os
import re

_NAME_BAD = re.compile(r"[^\w.-]")


def safe_name(name, default="item"):
    """A file or run name with everything but letters, digits, - _ . replaced, and no leading dots
    (so it can't become .., a hidden file or an absolute path)."""
    out = _NAME_BAD.sub("-", (name or "").strip()).strip(".")
    return out or default


def under(base, *parts):
    """Join parts under base and return the absolute path, or None if the result would leave base
    (.., absolute parts, Windows drive letters or UNC paths)."""
    base = os.path.abspath(base)
    for p in parts:
        p = str(p)
        if not p or p.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", p):
            return None
    target = os.path.abspath(os.path.join(base, *parts))
    try:
        return target if os.path.commonpath([target, base]) == base and target != base else None
    except ValueError:  # different drives on Windows
        return None


def inside(base, path):
    """True when path (an existing or not) lies inside base."""
    base, path = os.path.abspath(base), os.path.abspath(path)
    try:
        return os.path.commonpath([base, path]) == base
    except ValueError:
        return False
