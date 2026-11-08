"""Read the text out of a PDF, as well as the file allows.

  extract(path)  -> text      raises PdfError with a message meant for people
  check(path)    -> None | message   quick look (no text extraction), used by the Studio at upload

What it handles:
  * pypdf first, page by page: one broken page no longer loses the whole book;
  * PDFs locked with an empty password (most "protected" e-books), including AES (needs `cryptography`);
  * a poor or empty text layer: tries pdfminer.six, PyMuPDF and poppler's pdftotext when they are
    installed and keeps whichever reads best (Armenian, Cyrillic and other scripts included);
  * scanned pages: OCR with Tesseract when it is installed (pip install pytesseract + the Tesseract
    program); otherwise a clear message says how to get it;
  * running headers, footers and page numbers are dropped, and words hyphenated across lines re-joined;
  * a time budget per file, so one pathological PDF can't hold up a whole batch;
  * files that are not PDFs at all (an HTML error page saved as .pdf, an empty or cut-off download).
"""
import os, re, shutil, subprocess, sys, tempfile, time
from collections import Counter

MIN_PAGE_CHARS = 25          # fewer readable characters than this: the page has no usable text layer
DEFAULT_BUDGET = 900         # seconds per file for text extraction (OCR gets its own budget)
OCR_BUDGET = 3600            # seconds per file for OCR (about 2-6 s a page on a laptop CPU)
OCR_MAX_PAGES = 1000

TESSERACT_HINT = ("install Tesseract OCR to read scanned PDFs: `pip install pytesseract`, plus the Tesseract "
                  "program (Windows: https://github.com/UB-Mannheim/tesseract/wiki, macOS: `brew install "
                  "tesseract`, Linux: `apt install tesseract-ocr`; add language packs such as tesseract-ocr-hye "
                  "for Armenian), then try again. Or run `ocrmypdf in.pdf out.pdf` and upload out.pdf")


class PdfError(ValueError):
    """A PDF we can't read, with a message that says why and what to do."""


# ---------------------------------------------------------------- quick checks
def sniff(path):
    """None if the file starts like a PDF, otherwise why it isn't one."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"can't open the file ({e.strerror or e})"
    if size == 0:
        return "the file is empty (0 bytes); the upload or download didn't finish"
    with open(path, "rb") as f:
        head = f.read(2048)
    if b"%PDF-" in head[:1024]:
        return None
    low = head.lower()
    if b"<html" in low or b"<!doctype" in low:
        return "this is a web page saved with a .pdf name, not a PDF (the download probably failed: open the link in a browser and save the real file)"
    if head.startswith(b"PK"):
        return "this is a ZIP/Office file with a .pdf name; rename it to .docx (or unzip it) and add it again"
    return "this file is not a PDF (it doesn't start with %PDF); it may be damaged or another format renamed to .pdf"


def _open(path):
    """pypdf reader, unlocked when it only has an empty password. Raises PdfError for locked files."""
    import logging

    from pypdf import PdfReader
    from pypdf.errors import DependencyError, FileNotDecryptedError

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # "incorrect startxref" etc. are fixed silently
    reader = PdfReader(path, strict=False)
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except DependencyError:
            raise PdfError("this PDF is encrypted with AES; reading it needs the `cryptography` package: "
                           "run `pip install cryptography` (or run.bat again, which installs it)") from None
        except (FileNotDecryptedError, NotImplementedError) as e:
            raise PdfError(f"this PDF uses an encryption pypdf can't open ({e})") from None
        if not ok:
            raise PdfError("this PDF is protected by a password. Open it, then save or print it as a new PDF "
                           "without a password (or `qpdf --decrypt --password=... in.pdf out.pdf`) and add that")
    return reader


def check(path):
    """Fast look at a PDF right after upload: None when it can be opened, else the reason it can't.
    Does not extract text, so scanned PDFs pass (OCR may still read them)."""
    bad = sniff(path)
    if bad:
        return bad
    try:
        reader = _open(path)
        n = len(reader.pages)
    except PdfError as e:
        return str(e)
    except Exception as e:  # pypdf could not parse it at all: maybe another reader can
        if _other_readers():
            return None
        return f"the PDF is damaged and can't be read ({type(e).__name__}: {e})"
    if n == 0:
        return "the PDF has no pages"
    return None


# ---------------------------------------------------------------- text quality
_WORDISH = re.compile(r"[^\W\d_]", re.UNICODE)


def readable(text):
    """Count of letters, discounting signs of a broken text layer: (cid:12) codes, U+FFFD, private-use glyphs."""
    if not text:
        return 0
    bad = text.count("(cid:") * 6 + text.count("\ufffd") + sum(1 for ch in text if "\ue000" <= ch <= "\uf8ff")
    return max(0, len(_WORDISH.findall(text)) - 2 * bad)


def score(pages):
    return sum(readable(p) for p in pages)


# ---------------------------------------------------------------- extractors (each returns a list of pages)
def _pages_pypdf(path, deadline, log):
    reader = _open(path)
    pages, failed = [], 0
    for i, page in enumerate(reader.pages):
        if time.time() > deadline:
            log(f"  pypdf: stopped at page {i + 1} of {len(reader.pages)} (time budget used up)")
            break
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            failed += 1
            pages.append("")
    if failed:
        log(f"  pypdf: {failed} page(s) could not be read")
    return pages


def _pages_pdfminer(path, deadline, log):
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages = []
    for layout in extract_pages(path):
        if time.time() > deadline:
            break
        pages.append("".join(el.get_text() for el in layout if isinstance(el, LTTextContainer)))
    return pages


def _pages_pymupdf(path, deadline, log):
    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PdfError("password-protected")
        pages = []
        for page in doc:
            if time.time() > deadline:
                break
            pages.append(page.get_text())
    return pages


def _pages_pdftotext(path, deadline, log):
    exe = shutil.which("pdftotext")
    out = subprocess.run([exe, "-enc", "UTF-8", path, "-"], capture_output=True,
                         timeout=max(5, deadline - time.time()))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip()[:200])
    return out.stdout.decode("utf-8", "replace").split("\f")


def _importable(mod):
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def _other_readers():
    """The fallback extractors that are installed, best first."""
    out = []
    if _importable("fitz"):
        out.append(("PyMuPDF", _pages_pymupdf))
    if _importable("pdfminer"):
        out.append(("pdfminer", _pages_pdfminer))
    if shutil.which("pdftotext"):
        out.append(("pdftotext", _pages_pdftotext))
    return out


# ---------------------------------------------------------------- OCR
def tesseract_cmd():
    """Path of the Tesseract program, or None. Looks on PATH and in the usual Windows install folders."""
    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if exe:
        return exe
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", ""),
                     os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
            p = os.path.join(base, "Tesseract-OCR", "tesseract.exe")
            if base and os.path.isfile(p):
                return p
    return None


def ocr_available():
    return bool(tesseract_cmd()) and _importable("pytesseract") and _importable("PIL")


def ocr_languages(want="auto"):
    """Tesseract language codes joined with '+'. 'auto': English plus up to two other installed packs
    (Armenian and Russian first), so a mixed scan still reads."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    try:
        have = [l for l in pytesseract.get_languages(config="") if l not in ("osd", "equ", "snum")]
    except Exception:
        have = ["eng"]
    if want and want != "auto":
        langs = [l for l in re.split(r"[+,\s]+", want) if l]
        missing = [l for l in langs if l not in have]
        if missing:
            raise PdfError(f"Tesseract has no language pack for {', '.join(missing)} (installed: {', '.join(have) or 'none'})")
        return "+".join(langs)
    order = ["eng", "hye", "rus"] + sorted(have)
    langs = []
    for l in order:
        if l in have and l not in langs:
            langs.append(l)
    return "+".join(langs[:3]) or "eng"


def _page_images(path, numbers, tmp):
    """Yield (page number, PIL image) for the given 0-based pages, rendered at about 300 dpi."""
    from PIL import Image

    if _importable("fitz"):
        import fitz

        with fitz.open(path) as doc:
            doc.authenticate("")
            for n in numbers:
                pix = doc[n].get_pixmap(dpi=300)
                yield n, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return
    if shutil.which("pdftoppm"):
        for n in numbers:
            stem = os.path.join(tmp, f"p{n}")
            subprocess.run([shutil.which("pdftoppm"), "-r", "300", "-gray", "-png", "-f", str(n + 1), "-l", str(n + 1),
                            "-singlefile", path, stem], capture_output=True, timeout=300, check=True)
            img = Image.open(stem + ".png")
            img.load()
            os.remove(stem + ".png")
            yield n, img
        return
    # no renderer: scanned PDFs usually hold one picture per page, which pypdf can hand us directly
    reader = _open(path)
    for n in numbers:
        imgs = reader.pages[n].images
        if imgs:
            biggest = max(imgs, key=lambda im: im.image.width * im.image.height)
            yield n, biggest.image


def ocr_pages(path, numbers, lang="auto", budget=OCR_BUDGET, log=print):
    """OCR text for the given pages: {page number: text}."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    langs = ocr_languages(lang)
    numbers = list(numbers)[:OCR_MAX_PAGES]
    log(f"  OCR: reading {len(numbers)} scanned page(s) with Tesseract ({langs}); this takes a few seconds a page")
    out, start = {}, time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for k, (n, img) in enumerate(_page_images(path, numbers, tmp)):
            if time.time() - start > budget:
                log(f"  OCR: stopped after {k} page(s) (time budget used up)")
                break
            try:
                out[n] = pytesseract.image_to_string(img.convert("L"), lang=langs)
            except Exception as e:
                log(f"  OCR: page {n + 1} failed ({e})")
            if k and k % 25 == 0:
                log(f"  OCR: {k}/{len(numbers)} pages")
    return out


# ---------------------------------------------------------------- tidy up
_PAGE_NO = re.compile(r"^\s*(page\s*)?[-–—]?\s*(\d{1,4}|[ivxlcdm]{1,6})\s*[-–—]?\s*(of\s*\d+)?\s*$", re.I)


def tidy(pages):
    """Drop page numbers and running headers/footers (a short line repeated as the first or last line
    of many pages, digits ignored so "Chapter 3 · 41" matches "Chapter 3 · 42"), then join the pages."""
    split = [[l.rstrip() for l in p.replace("\r\n", "\n").replace("\r", "\n").split("\n")] for p in pages]

    def ends(lines):
        body = [i for i, l in enumerate(lines) if l.strip()]
        return {body[0], body[-1]} if body else set()

    key = lambda l: re.sub(r"\d+", "#", l.strip())
    edge = Counter(key(lines[i]) for lines in split for i in ends(lines))
    repeated = {k for k, c in edge.items() if len(split) >= 4 and c >= max(3, len(split) * 0.4) and len(k) <= 70}
    out = []
    for lines in split:
        drop = {i for i in ends(lines) if _PAGE_NO.match(lines[i]) or key(lines[i]) in repeated}
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop).strip())
    text = "\n\n".join(p for p in out if p)
    # words hyphenated across a line break, in any script: "mag-\nnificent" -> "magnificent"
    text = re.sub(r"([^\W\d_])-\n([^\W\d_])", lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), text)
    return text


# ---------------------------------------------------------------- main entry
def extract(path, ocr="auto", ocr_lang="auto", budget=DEFAULT_BUDGET, log=print):
    """Text of the PDF. ocr: 'auto' (OCR pages without text when Tesseract is installed) or 'off'."""
    bad = sniff(path)
    if bad:
        raise PdfError(bad)
    deadline = time.time() + budget
    candidates, errors = [], []
    try:
        candidates.append(("pypdf", _pages_pypdf(path, deadline, log)))
    except PdfError:
        raise
    except Exception as e:
        errors.append(f"pypdf: {type(e).__name__}: {e}")

    def good_enough(pages):
        return pages and sum(1 for p in pages if readable(p) >= MIN_PAGE_CHARS) >= 0.9 * len(pages)

    if not (candidates and good_enough(candidates[0][1])):
        for name, fn in _other_readers():
            if time.time() > deadline:
                break
            try:
                pages = fn(path, deadline, log)
                candidates.append((name, pages))
                if good_enough(pages):
                    break
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
    if not candidates:
        hint = "" if _other_readers() else " (installing `pdfminer.six` gives a second reader that copes with more files)"
        raise PdfError("the PDF is damaged and can't be read: " + "; ".join(errors)[:400] + hint)

    name, pages = max(candidates, key=lambda c: score(c[1]))
    if len(candidates) > 1:
        log(f"  read with {name} (best of {', '.join(c[0] for c in candidates)})")

    empty = [i for i, p in enumerate(pages) if readable(p) < MIN_PAGE_CHARS]
    if empty and ocr != "off" and len(empty) >= max(1, 0.2 * len(pages)):
        if ocr_available():
            for n, t in ocr_pages(path, empty, ocr_lang, log=log).items():
                if readable(t) > readable(pages[n]):
                    pages[n] = t
        elif score(pages) < 200:
            raise PdfError("no text layer found: this looks like a scanned PDF (pictures of pages). To read it, "
                           + TESSERACT_HINT)
        else:
            log(f"  {len(empty)} page(s) look scanned and were skipped; to read them, {TESSERACT_HINT}")

    text = tidy(pages)
    if readable(text) < 50:
        if ocr == "off" and empty:
            raise PdfError("no text layer found (scanned PDF?) and OCR is turned off")
        raise PdfError("almost no readable text in this PDF (it may be scanned, or use fonts that don't map to letters)"
                       + ("" if ocr_available() else "; " + TESSERACT_HINT))
    return text


if __name__ == "__main__":  # python data/pdftext.py file.pdf  -> prints the text
    print(extract(sys.argv[1], log=lambda m: print(m, file=sys.stderr)))
