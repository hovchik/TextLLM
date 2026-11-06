"""Guess the language of a text. Returns an ISO 639-1 code ("en", "hy", "ru", ...) or None.

Uses the `langdetect` package when it is installed (pip install langdetect). Without it, a
small built-in guesser looks at the script (Armenian, Cyrillic, Greek, ...) and, for Latin and
Cyrillic scripts, at the most common short words of a dozen languages.
"""
import re
import unicodedata

_STOP = {
    "en": "the and of to in is that it for was with as on be are this by at from or have not",
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch",
    "fr": "le la les de des et un une du en est que qui dans pour pas sur au plus avec ce",
    "es": "el la de que y en los del se las por un para con una su al es lo como más",
    "it": "il di che la e è un per una non in sono con del le si della al da come",
    "pt": "de a o que e do da em um para com não uma os no se na por mais as dos",
    "nl": "de het een en van in is dat op te zijn voor met die niet aan er ook als",
    "pl": "i w z na się nie że jest do to jak o co po ale za od tak jego przez",
    "tr": "ve bir bu da de için ile olarak en çok daha gibi var olan sonra kadar ama",
    "sv": "och att det i en som är av för på med till den har inte om han var ett",
    "ru": "и в не на что с он как это по но его из за то все она так же от к",
    "uk": "і в не на що з він як це по але його із за то все вона так же від до",
    "bg": "и в не на че с той как това по но него от за то всички тя така също до",
}
_WORDS = {k: set(v.split()) for k, v in _STOP.items()}
_LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "sv"}
_CYRILLIC = {"ru", "uk", "bg"}
_SCRIPTS = [  # unicode block → language when the script belongs to (mostly) one language
    ((0x0530, 0x058F), "hy"), ((0x10A0, 0x10FF), "ka"), ((0x0370, 0x03FF), "el"), ((0x0590, 0x05FF), "he"),
    ((0x0E00, 0x0E7F), "th"), ((0x0900, 0x097F), "hi"), ((0xAC00, 0xD7AF), "ko"), ((0x3040, 0x30FF), "ja"),
    ((0x0600, 0x06FF), "ar"), ((0x4E00, 0x9FFF), "zh"), ((0x0400, 0x04FF), "cyrillic"),
]


def _script(text):
    """(dominant script, letters seen). Script is a language code for single-language scripts,
    or "latin" / "cyrillic"."""
    sample = text[:20000]
    counts, letters = {}, 0
    for ch in sample:
        o = ord(ch)
        if o < 0x250:
            if ch.isalpha():
                letters += 1
                counts["latin"] = counts.get("latin", 0) + 1
            continue
        if not unicodedata.category(ch).startswith("L"):
            continue
        letters += 1
        for (lo, hi), lang in _SCRIPTS:
            if lo <= o <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if letters < 20 or not counts:
        return None, letters
    cjk = counts.get("zh", 0) + counts.get("ja", 0)
    if cjk:  # kanji and kana mix: any real amount of kana means Japanese
        counts["zh"], counts["ja"] = (0, cjk) if counts.get("ja", 0) > cjk * 0.1 else (cjk, 0)
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return (script if n >= letters * 0.5 else None), letters


def _by_words(text, script):
    words = re.findall(r"[^\W\d_]+", text[:20000].lower())
    if not words:
        return None
    cands = _LATIN if script == "latin" else _CYRILLIC
    scores = {k: sum(1 for w in words if w in _WORDS[k]) for k in cands}
    best, hits = max(scores.items(), key=lambda kv: kv[1])
    return best if hits >= max(3, len(words) * 0.02) else None


def detect(text):
    if not text or len(text.strip()) < 40:
        return None
    script, _ = _script(text)
    if script is None:
        return None
    if script not in ("latin", "cyrillic"):
        return script  # one language per script (langdetect doesn't know Armenian or Georgian)
    try:
        from langdetect import DetectorFactory, detect as _detect

        DetectorFactory.seed = 0
        code = (_detect(text[:20000]) or "").split("-")[0]
        if code and (code in _CYRILLIC) == (script == "cyrillic"):
            return code
    except Exception:
        pass
    return _by_words(text, script)


NAMES = {"en": "English", "hy": "Armenian", "ru": "Russian", "uk": "Ukrainian", "de": "German", "fr": "French",
         "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
         "sv": "Swedish", "bg": "Bulgarian", "ka": "Georgian", "el": "Greek", "he": "Hebrew", "ar": "Arabic",
         "fa": "Persian", "th": "Thai", "hi": "Hindi", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}


def name(code):
    return NAMES.get(code, code or "unknown")
