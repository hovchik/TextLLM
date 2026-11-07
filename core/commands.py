"""Check a saved training command before it is run again.

Runs keep their training settings in <run>/train_cmd.json, and Resume replays them. A run folder
can come from another PC (transfer.py import), so the saved command is checked: only the project's
own training scripts, only well-formed arguments, and every path argument inside the project.
"""
import os
import re

from core import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {"train.py", "finetune/finetune_lora.py", "cluster.py"}
# values of these flags are paths and must stay inside the project folder
PATH_FLAGS = {"--out", "--data", "--instructions", "--extra", "--raw", "--code", "--init-adapter", "--tok"}
_ARG_OK = re.compile(r"^[^\x00-\x1f\x7f]{1,4000}$")  # printable, one line (arguments never go through a shell)


class BadCommand(ValueError):
    pass


def check(cmd):
    """cmd: the list saved in train_cmd.json (script first, then its arguments). Returns it
    unchanged, or raises BadCommand."""
    if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
        raise BadCommand("train_cmd.json must be a list of strings")
    if cmd[0].replace("\\", "/") not in SCRIPTS:
        raise BadCommand(f"train_cmd.json names {cmd[0]!r}, which is not one of the training scripts")
    for prev, c in zip(cmd, cmd[1:]):
        if not _ARG_OK.match(c):
            raise BadCommand(f"unexpected argument in train_cmd.json: {c[:60]!r}")
        if prev in PATH_FLAGS:
            if c.startswith("-"):
                raise BadCommand(f"{prev} needs a path, got {c!r}")
            for part in c.split(","):  # --raw takes several folders
                if not paths.inside(ROOT, os.path.expanduser(os.path.join(ROOT, part))):
                    raise BadCommand(f"{prev} {part!r} points outside the project folder")
    return cmd
