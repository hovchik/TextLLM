"""Turn your own files and web links into training text.

Supported: .pdf, .docx, .doc, .txt, .md, .html, and URLs (web articles, online PDFs, plain-text books).

  python data/ingest.py --files book.pdf notes.md --urls https://example.com/article --out data/raw/custom
  python data/ingest.py --files *.pdf --delete-originals      # remove source files after extracting

Output is a shard of documents separated by <|endoftext|>, the same format as build_corpus.py,
so the tokenizer, train.py and finetune_lora.py all pick it up from data/raw/.
"""
import argparse, hashlib, os, re, shutil, subprocess, sys, tempfile, time

EOT = "<|endoftext|>"
TEXT_EXT = {".txt", ".md", ".markdown", ".text"}
UA = {"User-Agent": "text-llm-ingest/1.0"}


def clean(text):
    # Strip Project Gutenberg boilerplate when present.
    m = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I | re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
    if m:
        text = text[: m.start()]
    text = text.replace("\r\n", "\n").replace("\x00", "").replace(EOT, "")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join words hyphenated across PDF lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def from_pdf(path):
    from pypdf import PdfReader

    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    text = "\n\n".join(pages)
    if len(text.strip()) < 50:
        raise ValueError("no text layer found (scanned PDF?). Run OCR first, e.g. `ocrmypdf in.pdf out.pdf`")
    return text


def from_docx(path):
    import docx

    d = docx.Document(path)
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


def from_doc(path):
    """Old Word .doc: try antiword, then LibreOffice."""
    if shutil.which("antiword"):
        return subprocess.run(["antiword", path], capture_output=True, text=True, check=True).stdout
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([office, "--headless", "--convert-to", "txt:Text", "--outdir", td, path],
                           capture_output=True, check=True, timeout=300)
            out = os.path.join(td, os.path.splitext(os.path.basename(path))[0] + ".txt")
            return open(out, encoding="utf-8", errors="replace").read()
    raise ValueError("reading .doc needs antiword or LibreOffice installed (or save the file as .docx)")


def from_html(html, url=None):
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_recall=True)
        if text and len(text) > 200:
            return text
    except Exception:
        pass
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        t.decompose()
    return soup.get_text("\n")


def from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return from_pdf(path)
    if ext == ".docx":
        return from_docx(path)
    if ext == ".doc":
        return from_doc(path)
    if ext in (".html", ".htm"):
        return from_html(open(path, encoding="utf-8", errors="replace").read())
    if ext in TEXT_EXT or ext == "":
        raw = open(path, "rb").read()
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    raise ValueError(f"unsupported file type {ext}")


def from_url(url, tmpdir):
    """Download to a temp file, extract, and leave deletion to the caller (tmpdir is removed)."""
    from core import net

    r = net.get(url, timeout=60)  # refuses local-network addresses, caps the size
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    name = os.path.basename(url.split("?")[0]) or "page"
    if "pdf" in ctype or name.lower().endswith(".pdf"):
        path = os.path.join(tmpdir, hashlib.md5(url.encode()).hexdigest() + ".pdf")
        open(path, "wb").write(r.content)
        return from_pdf(path)
    if name.lower().endswith(".docx") or "officedocument.wordprocessingml" in ctype:
        path = os.path.join(tmpdir, hashlib.md5(url.encode()).hexdigest() + ".docx")
        open(path, "wb").write(r.content)
        return from_docx(path)
    if "html" in ctype or "<html" in r.text[:2000].lower():
        return from_html(r.text, url)
    r.encoding = r.encoding or r.apparent_encoding or "utf-8"
    return r.text


SUPPORTED = {".pdf", ".docx", ".doc", ".html", ".htm"} | TEXT_EXT


def expand(paths):
    """Folders are searched recursively for supported files."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for d, _, fs in os.walk(p):
                out += [os.path.join(d, f) for f in sorted(fs) if os.path.splitext(f)[1].lower() in SUPPORTED]
        else:
            out.append(p)
    return out


def ingest(files=(), urls=(), out="data/raw/custom", delete_originals=False, log=print):
    os.makedirs(out, exist_ok=True)
    files = expand(files)
    shard = os.path.join(out, f"shard_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    ok, chars = 0, 0
    with open(shard, "w", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
        sources = [("file", p) for p in files] + [("url", u.strip()) for u in urls if u.strip()]
        for kind, src in sources:
            try:
                text = clean(from_file(src) if kind == "file" else from_url(src, tmp))
                if len(text) < 200:
                    raise ValueError("almost no text extracted")
                f.write(text + f"\n{EOT}\n")
                ok, chars = ok + 1, chars + len(text)
                log(f"✓ {src}: {len(text):,} characters")
                if kind == "file" and delete_originals:
                    os.remove(src)
                    log(f"  removed original {src}")
            except Exception as e:
                log(f"✗ {src}: {e}")
    if ok == 0:
        os.remove(shard)
        return None, 0, 0
    log(f"wrote {shard} ({ok} documents, {chars:,} characters, about {chars // 4:,} tokens)")
    return shard, ok, chars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--urls", nargs="*", default=[])
    ap.add_argument("--url-list", help="text file with one URL per line")
    ap.add_argument("--out", default="data/raw/custom")
    ap.add_argument("--delete-originals", action="store_true")
    a = ap.parse_args()
    urls = list(a.urls) + ([l for l in open(a.url_list) if l.strip()] if a.url_list else [])
    shard, n, _ = ingest(a.files, urls, a.out, a.delete_originals, log=lambda m: print(m, flush=True))
    sys.exit(0 if shard else 1)
