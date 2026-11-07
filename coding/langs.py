"""Which files count as source code, their language, and the one-line header each document starts with.

Every code document in data/code/ starts with a header line in the file's own comment syntax, e.g.

    # path: myproject/app/models.py
    // path: lodash/src/chunk.ts

so the model learns which language and file it is in, and other tools can tell a code document from
prose (see is_code_doc)."""
import os, re

# extension -> (language, line comment or (open, close) block comment)
EXT = {
    ".py": ("Python", "#"), ".pyi": ("Python", "#"), ".ipynb": ("Python", "#"),
    ".js": ("JavaScript", "//"), ".mjs": ("JavaScript", "//"), ".cjs": ("JavaScript", "//"),
    ".jsx": ("JavaScript", "//"), ".ts": ("TypeScript", "//"), ".tsx": ("TypeScript", "//"),
    ".java": ("Java", "//"), ".kt": ("Kotlin", "//"), ".kts": ("Kotlin", "//"), ".scala": ("Scala", "//"),
    ".groovy": ("Groovy", "//"), ".gradle": ("Groovy", "//"),
    ".go": ("Go", "//"), ".rs": ("Rust", "//"), ".swift": ("Swift", "//"), ".dart": ("Dart", "//"),
    ".c": ("C", "//"), ".h": ("C", "//"), ".cpp": ("C++", "//"), ".cc": ("C++", "//"), ".cxx": ("C++", "//"),
    ".hpp": ("C++", "//"), ".hh": ("C++", "//"), ".hxx": ("C++", "//"), ".m": ("Objective-C", "//"),
    ".mm": ("Objective-C", "//"), ".cs": ("C#", "//"), ".fs": ("F#", "//"), ".vb": ("Visual Basic", "'"),
    ".php": ("PHP", "//"), ".rb": ("Ruby", "#"), ".pl": ("Perl", "#"), ".pm": ("Perl", "#"),
    ".lua": ("Lua", "--"), ".r": ("R", "#"), ".jl": ("Julia", "#"), ".ex": ("Elixir", "#"), ".exs": ("Elixir", "#"),
    ".erl": ("Erlang", "%"), ".hs": ("Haskell", "--"), ".ml": ("OCaml", ("(*", "*)")), ".clj": ("Clojure", ";"),
    ".elm": ("Elm", "--"), ".nim": ("Nim", "#"), ".zig": ("Zig", "//"), ".v": ("Verilog", "//"),
    ".sv": ("SystemVerilog", "//"), ".vhd": ("VHDL", "--"), ".f90": ("Fortran", "!"),
    ".sh": ("Shell", "#"), ".bash": ("Shell", "#"), ".zsh": ("Shell", "#"), ".fish": ("Shell", "#"),
    ".ps1": ("PowerShell", "#"), ".psm1": ("PowerShell", "#"), ".bat": ("Batchfile", "REM"), ".cmd": ("Batchfile", "REM"),
    ".sql": ("SQL", "--"), ".html": ("HTML", ("<!--", "-->")), ".htm": ("HTML", ("<!--", "-->")),
    ".vue": ("Vue", ("<!--", "-->")), ".svelte": ("Svelte", ("<!--", "-->")), ".xml": ("XML", ("<!--", "-->")),
    ".css": ("CSS", ("/*", "*/")), ".scss": ("SCSS", "//"), ".sass": ("Sass", "//"), ".less": ("Less", "//"),
    ".yaml": ("YAML", "#"), ".yml": ("YAML", "#"), ".toml": ("TOML", "#"), ".ini": ("INI", ";"),
    ".cmake": ("CMake", "#"), ".mk": ("Makefile", "#"), ".tf": ("HCL", "#"), ".hcl": ("HCL", "#"),
    ".proto": ("Protocol Buffer", "//"), ".graphql": ("GraphQL", "#"), ".gql": ("GraphQL", "#"),
    ".sol": ("Solidity", "//"), ".asm": ("Assembly", ";"), ".s": ("Assembly", "#"),
    ".md": ("Markdown", ("<!--", "-->")), ".rst": ("reStructuredText", ".."),
}
NAMES = {"dockerfile": ("Dockerfile", "#"), "makefile": ("Makefile", "#"), "cmakelists.txt": ("CMake", "#"),
         "gemfile": ("Ruby", "#"), "rakefile": ("Ruby", "#"), "jenkinsfile": ("Groovy", "//"),
         "vagrantfile": ("Ruby", "#")}
SHEBANG = [(re.compile(r"python"), "Python", "#"), (re.compile(r"\b(ba|z|da|k)?sh\b"), "Shell", "#"),
           (re.compile(r"node|deno|bun"), "JavaScript", "//"), (re.compile(r"ruby"), "Ruby", "#"),
           (re.compile(r"perl"), "Perl", "#"), (re.compile(r"php"), "PHP", "//")]
# Languages that are mostly prose or data: off unless asked for (--langs), so a copilot learns code.
DOCS = {"Markdown", "reStructuredText"}
FENCE = {"C++": "cpp", "C#": "csharp", "F#": "fsharp", "Objective-C": "objectivec", "Visual Basic": "vb",
         "Protocol Buffer": "protobuf", "Batchfile": "bat", "Shell": "bash"}

HEADER_RE = re.compile(r"^(#|//|--|;|%|!|'|REM|\.\.|<!--|/\*|\(\*) path: (\S[^\n]*?)( -->| \*/| \*\))?$")


def detect(path, head=""):
    """(language, comment) for a file path, or None if it isn't a known source file.
    head: the start of the file, used for extensionless scripts with a #! line."""
    base = os.path.basename(path).lower()
    if base in NAMES:
        return NAMES[base]
    if base.startswith("dockerfile"):
        return NAMES["dockerfile"]
    ext = os.path.splitext(base)[1]
    if ext in EXT:
        return EXT[ext]
    if not ext and head.startswith("#!"):
        line = head.split("\n", 1)[0]
        for rx, lang, com in SHEBANG:
            if rx.search(line):
                return lang, com
    return None


def norm(name):
    """Compare language names loosely: 'c++' == 'C++', 'javascript' == 'JavaScript'."""
    return re.sub(r"\s+", "", (name or "").lower())


def wanted(lang, langs):
    """langs: set of normalised names to keep, or empty = every code language (not Markdown/RST)."""
    return norm(lang) in langs if langs else lang not in DOCS


def parse_langs(s):
    return {norm(x) for x in re.split(r"[,;]", s or "") if x.strip()}


def header(path, comment):
    path = path.replace("\\", "/")
    if isinstance(comment, tuple):
        return f"{comment[0]} path: {path} {comment[1]}"
    return f"{comment} path: {path}"


def is_code_doc(doc):
    return bool(HEADER_RE.match(doc.split("\n", 1)[0]))


def doc_path(doc):
    m = HEADER_RE.match(doc.split("\n", 1)[0])
    return m.group(2).strip() if m else None


def doc_lang(doc):
    p = doc_path(doc)
    d = detect(p) if p else None
    return d[0] if d else None


def fence(lang):
    """The Markdown code-fence name for a language (```python)."""
    return FENCE.get(lang, norm(lang or ""))
