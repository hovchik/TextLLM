"""The core: small, dependency-light pieces every path (Studio, scripts, cluster, copilot) shares.

    core.safeload   load checkpoints and tokenized examples as data only (never pickled code)
    core.net        fetch URLs without reaching into the local network (SSRF guard, size caps)
    core.paths      keep user-supplied names and paths inside the folders they belong to
    core.commands   check a saved training command before it is run again
    core.backends   which fast (C++/Rust) engines are installed: llama.cpp, tokenizers, faiss, ...
    core.bench      micro-benchmarks for the hot paths
    core.quiet      mutes harmless library warnings (e.g. "triton not found" on Windows)

Nothing here imports torch at module level, so the Studio starts fast.
"""

from . import quiet  # noqa: E402,F401  (installs the log filters on any `import core...`)
