"""A small .gitignore matcher (the common rules: globs, **, anchors, dir/, !negation, nested files).

    rules = Ignore()
    rules.add_file("/proj/.gitignore", base="")          # base: folder of that file, relative to the root
    rules.ignored("build/out.js", is_dir=False)
"""
import re


def _translate(pat):
    """gitignore glob -> regex for a path relative to the .gitignore's folder."""
    i, n, out = 0, len(pat), ""
    while i < n:
        c = pat[i]
        if pat.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pat.startswith("/**", i) and i + 3 == n:
            out += "/.*"
            i += 3
        elif pat.startswith("**", i):
            out += ".*"
            i += 2
        elif c == "*":
            out += "[^/]*"
            i += 1
        elif c == "?":
            out += "[^/]"
            i += 1
        elif c == "[":
            j = pat.find("]", i + 2 if i + 1 < n and pat[i + 1] in "!^" else i + 1)
            if j == -1:
                out += re.escape(c)
                i += 1
            else:
                body = pat[i + 1:j]
                if body[:1] in "!^":
                    body = "^" + body[1:]
                out += "[" + body.replace("\\", "\\\\") + "]"
                i = j + 1
        elif c == "\\" and i + 1 < n:
            out += re.escape(pat[i + 1])
            i += 2
        else:
            out += re.escape(c)
            i += 1
    return re.compile(out + r"\Z")


class Ignore:
    def __init__(self, patterns=(), base=""):
        self.rules = []  # (base, regex, negate, dir_only, anchored)
        self.add_lines(patterns, base)

    def add_file(self, path, base=""):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                self.add_lines(f.read().splitlines(), base)
        except OSError:
            pass

    def add_lines(self, lines, base=""):
        base = base.strip("/")
        for line in lines:
            line = line.rstrip("\n\r")
            if not line.strip() or line.startswith("#"):
                continue
            if not line.endswith("\\ "):
                line = line.rstrip()
            neg = line.startswith("!")
            if neg:
                line = line[1:]
            elif line.startswith("\\!") or line.startswith("\\#"):
                line = line[1:]
            dir_only = line.endswith("/")
            line = line.rstrip("/")
            if not line:
                continue
            anchored = "/" in line
            line = line.lstrip("/")
            self.rules.append((base, _translate(line), neg, dir_only, anchored))

    def ignored(self, rel, is_dir=False):
        rel = rel.replace("\\", "/").strip("/")
        name = rel.rsplit("/", 1)[-1]
        hit = False
        for base, rx, neg, dir_only, anchored in self.rules:
            if dir_only and not is_dir:
                continue
            if base:
                if not rel.startswith(base + "/"):
                    continue
                sub = rel[len(base) + 1:]
            else:
                sub = rel
            if rx.match(sub if anchored else name):
                hit = not neg
        return hit
