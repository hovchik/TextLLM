"""Fill-in-the-middle (FIM): teach a model to write the code between a prefix and a suffix.

A document is cut into prefix / middle / suffix and rearranged as (PSM order, as in Qwen2.5-Coder,
StarCoder and Code Llama):

    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>

so the model reads what comes before and after the cursor and learns to write the missing middle.
The same token names are used by our from-scratch tokenizer (train_tokenizer.py --code) and by
Qwen2.5-Coder / Qwen3, so prompts look the same for both kinds of model."""
import random

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FIM_PAD = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>"
EOT = "<|endoftext|>"
FILE_SEP = "<|file_sep|>"
SPECIAL = (EOT, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD)
CURSOR = "<cursor>"  # where Chat → Complete code puts the completion


def defuse(text):
    """Break up special-token strings that appear inside a source file (a tokenizer test, a file
    about LLMs), so they can't end a document early or look like FIM markers."""
    for t in SPECIAL + (FILE_SEP,):
        if t in text:
            text = text.replace(t, t.replace("<|", "<| ").replace("|>", " |>"))
    return text


def split(doc, rng=random, line_share=0.5, max_middle=None):
    """Cut a code document into (prefix, middle, suffix). The header line stays in the prefix.

    Half of the cuts (line_share) follow line boundaries, like "complete the next few lines"; the
    rest cut at random characters, like "finish this expression". max_middle caps the middle's length
    in characters."""
    head, nl, body = doc.partition("\n")
    head += nl
    if len(body) < 4:
        return doc, "", ""
    if rng.random() < line_share:
        lines = body.splitlines(keepends=True)
        i = rng.randrange(0, len(lines))
        j = rng.randrange(i, min(len(lines), i + rng.randint(1, 12))) + 1
        a = sum(map(len, lines[:i]))
        b = a + sum(map(len, lines[i:j]))
    else:
        a, b = sorted(rng.randrange(0, len(body) + 1) for _ in range(2))
    if max_middle and b - a > max_middle:
        b = a + max_middle
    return head + body[:a], body[a:b], body[b:]


def psm(prefix, suffix, middle=""):
    """The FIM prompt (and, with middle, the training text without the end token)."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


def token_ids(tok):
    """{prefix, suffix, middle} token ids if a tokenizer knows the FIM tokens, else None.
    Works with a `tokenizers.Tokenizer` and with a transformers tokenizer."""
    out = {}
    for k, t in (("prefix", FIM_PREFIX), ("suffix", FIM_SUFFIX), ("middle", FIM_MIDDLE)):
        if hasattr(tok, "token_to_id"):
            i = tok.token_to_id(t)
        else:
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i == getattr(tok, "unk_token_id", None):
                i = None
        if i is None:
            return None
        out[k] = i
    return out


def end_token(tok):
    """The token that ends a raw completion: <|endoftext|> if the tokenizer has it (Qwen, ours),
    otherwise the tokenizer's end-of-sequence token."""
    i = tok.convert_tokens_to_ids(EOT) if hasattr(tok, "convert_tokens_to_ids") else None
    if i is not None and i != getattr(tok, "unk_token_id", None):
        return EOT
    return tok.eos_token or ""


def split_cursor(text):
    """Chat → Complete code: the message is code with an optional <cursor> marker.
    Returns (prefix, suffix); without a marker the completion continues the end of the code."""
    if CURSOR in text:
        a, _, b = text.partition(CURSOR)
        return a, b
    return text, ""
