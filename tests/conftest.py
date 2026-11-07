import os, subprocess, sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EOT = "<|endoftext|>"
WORDS = ("the quiet harbour town woke slowly as fishing boats returned with nets full of silver light "
         "and children ran along the stone wall calling to gulls while old sailors told stories").split()


def sample_text(n_docs=40, seed=0):
    import random

    r = random.Random(seed)
    return [" ".join(r.choice(WORDS) for _ in range(r.randint(40, 120))) + "." for _ in range(n_docs)]


def run_script(*args, timeout=600):
    """Run one of the repo's scripts like the Studio does; fail the test with its output on error."""
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="session")
def bpe(tmp_path_factory):
    """A small tokenizer made by tokenizer/train_tokenizer.py itself, saved to a file."""
    from tokenizer.train_tokenizer import train

    d = tmp_path_factory.mktemp("tok")
    (d / "raw").mkdir()
    (d / "raw" / "shard.txt").write_text(EOT.join(sample_text(200)), encoding="utf-8")
    p = str(d / "bpe.json")
    train(str(d / "raw"), 400, p, 0)
    return p
