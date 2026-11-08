"""PDF reading for Your texts: broken, locked, scanned and non-Latin PDFs, and the upload check."""
import importlib, os, subprocess, sys

import pytest

from conftest import ROOT

from data import pdftext
from data.pdftext import PdfError

STORY = ("Once upon a time a small fox lived at the edge of a deep forest and listened to the rain. "
         "Every evening she walked to the river to watch the boats go home.")


def make_pdf(path, pages, header=None):
    """A real PDF with a text layer, written by hand (Helvetica, ASCII), one string list per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for n, lines in enumerate(pages):
        lines = ([header] if header else []) + list(lines) + [str(n + 1)]
        ops = "BT /F1 11 Tf 14 TL 40 800 Td " + " ".join(
            "(" + l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*" for l in lines) + " ET"
        objs.append(f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)
    return str(path)


def book(tmp_path, n=6, name="book.pdf"):
    return make_pdf(tmp_path / name, [[STORY[:90], STORY[90:], f"Page text number {i} goes on here."] for i in range(n)],
                    header="THE FOX AND THE RIVER")


def test_reads_text_and_drops_headers_and_page_numbers(tmp_path):
    text = pdftext.extract(book(tmp_path), log=lambda m: None)
    assert "small fox" in text and "Page text number 5" in text
    assert "THE FOX AND THE RIVER" not in text          # running header
    assert not any(l.strip().isdigit() for l in text.splitlines())  # page numbers


def test_one_broken_page_does_not_lose_the_book(tmp_path, monkeypatch):
    from pypdf import PageObject

    real, calls = PageObject.extract_text, {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyError("/Widths")  # what pypdf raises on some odd fonts
        return real(self, *a, **k)

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    monkeypatch.setattr(pdftext, "_other_readers", lambda: [])
    text = pdftext.extract(book(tmp_path), log=lambda m: None)
    assert "Page text number 0" in text and "Page text number 5" in text


def test_falls_back_to_another_reader_when_pypdf_gets_nothing(tmp_path, monkeypatch):
    pytest.importorskip("pdfminer")
    p = book(tmp_path)
    monkeypatch.setattr(pdftext, "_pages_pypdf", lambda path, deadline, log: [""] * 6)
    monkeypatch.setattr(pdftext, "ocr_available", lambda: False)
    monkeypatch.setattr(pdftext, "_other_readers", lambda: [("pdfminer", pdftext._pages_pdfminer)])
    logs = []
    text = pdftext.extract(p, log=logs.append)
    assert "small fox" in text and any("pdfminer" in l for l in logs)


@pytest.mark.parametrize("content, words", [
    (b"", "empty"),
    (b"<!DOCTYPE html><html><body>404 Not Found</body></html>", "web page"),
    (b"PK\x03\x04 zip bytes" * 10, "ZIP"),
    (b"just some bytes, not a pdf" * 20, "not a PDF"),
])
def test_files_that_are_not_pdfs_get_a_clear_message(tmp_path, content, words):
    p = tmp_path / "x.pdf"
    p.write_bytes(content)
    assert words in pdftext.check(str(p))
    with pytest.raises(PdfError, match=words):
        pdftext.extract(str(p))


def test_encrypted_pdfs(tmp_path):
    pytest.importorskip("cryptography")
    from pypdf import PdfWriter

    src = book(tmp_path)
    for name, user_pw in (("open.pdf", ""), ("locked.pdf", "secret")):
        w = PdfWriter(clone_from=src)
        w.encrypt(user_password=user_pw, owner_password="owner", algorithm="AES-256")
        w.write(str(tmp_path / name))
    # an empty user password (copy/print restrictions only) opens as normal
    assert pdftext.check(str(tmp_path / "open.pdf")) is None
    assert "small fox" in pdftext.extract(str(tmp_path / "open.pdf"), log=lambda m: None)
    # a real password is refused with a message that says what to do
    assert "password" in pdftext.check(str(tmp_path / "locked.pdf"))
    with pytest.raises(PdfError, match="password"):
        pdftext.extract(str(tmp_path / "locked.pdf"))


def test_damaged_pdf_explains_instead_of_crashing(tmp_path, monkeypatch):
    p = tmp_path / "cut.pdf"
    p.write_bytes(open(book(tmp_path), "rb").read()[:300])
    monkeypatch.setattr(pdftext, "_other_readers", lambda: [])
    with pytest.raises(PdfError, match="damaged"):
        pdftext.extract(str(p))


def test_scanned_pdf_without_ocr_says_how_to_get_it(tmp_path, monkeypatch):
    p = make_pdf(tmp_path / "scan.pdf", [[] for _ in range(3)])  # pages with only a page number
    monkeypatch.setattr(pdftext, "ocr_available", lambda: False)
    monkeypatch.setattr(pdftext, "_other_readers", lambda: [])
    with pytest.raises(PdfError, match="Tesseract"):
        pdftext.extract(p)


def test_scanned_pdf_is_read_with_ocr(tmp_path):
    if not pdftext.ocr_available():
        pytest.skip("Tesseract OCR is not installed")
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    for i, line in enumerate(["The small fox walked to the river", "to watch the boats go home again.",
                              "Every evening the rain fell softly."] * 3):
        d.text((100, 150 + i * 80), line, fill="black", font=font)
    p = str(tmp_path / "scan.pdf")
    img.save(p)
    text = pdftext.extract(p, ocr_lang="eng", log=lambda m: None)
    assert "river" in text and "boats" in text
    with pytest.raises(PdfError):
        pdftext.extract(p, ocr="off")


def test_armenian_and_cyrillic_text(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    try:
        pdfmetrics.registerFont(TTFont("DV", "DejaVuSans.ttf"))
    except Exception:
        pytest.skip("no DejaVu font")
    p = str(tmp_path / "hy.pdf")
    c = canvas.Canvas(p)
    c.setFont("DV", 12)
    c.drawString(40, 800, "Մի անգամ մի փոքրիկ աղվես էր ապրում խիտ անտառի եզրին։")
    c.drawString(40, 780, "Жила-была маленькая лиса на краю большого тёмного леса.")
    c.save()
    text = pdftext.extract(p, log=lambda m: None)
    assert "աղվես" in text and "лиса" in text


def test_hyphenation_rejoined_in_any_script():
    from data.ingest import clean

    assert clean("a mag-\nnificent fox") == "a magnificent fox"
    assert clean("փոքրիկ աղ-\nվես") == "փոքրիկ աղվես"
    assert clean("New-\nYork") == "New-\nYork"  # a capital after the break: probably a real hyphen


def test_ingest_survives_a_windows_code_page_stdout(tmp_path):
    """On Windows a redirected stdout is cp1252, which can't print ✓/✗: every upload used to crash."""
    p = book(tmp_path, name="Աղվես.pdf")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    r = subprocess.run([sys.executable, "data/ingest.py", "--files", p, "--out", str(tmp_path / "out")],
                       cwd=ROOT, capture_output=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    assert "✓" in r.stdout.decode("utf-8")
    shard = os.listdir(tmp_path / "out")[0]
    assert "small fox" in open(tmp_path / "out" / shard, encoding="utf-8").read()


def test_ingest_reports_every_file_and_keeps_the_good_ones(tmp_path):
    from data.ingest import ingest

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"<html>error</html>")
    logs = []
    shard, n, _ = ingest([book(tmp_path), str(bad), str(tmp_path / "missing.pdf")], out=str(tmp_path / "o"),
                         log=logs.append)
    assert shard and n == 1
    assert any(l.startswith("✗") and "web page" in l for l in logs)
    assert any(l.startswith("✗") and "not found" in l for l in logs)


def test_studio_upload_checks_pdfs_before_starting(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app
    from studio import auth, db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "studio.db"))
    db.init()
    importlib.reload(auth)
    launched = []
    monkeypatch.setattr(app, "_launch", lambda *a, **k: launched.append((a, k)) or {"running": True})
    c = TestClient(app.app)
    c.post("/api/auth/setup", json={"username": "admin", "password": "secret123"})
    good = open(book(tmp_path), "rb").read()
    # only a broken file: refused at once, with the reason
    r = c.post("/api/texts", data={"action": "add"}, files=[("files", ("թուղթ.pdf", b"<html>404</html>", "application/pdf"))])
    assert r.status_code == 400 and "թուղթ.pdf" in r.json()["detail"] and "web page" in r.json()["detail"]
    assert not launched
    # a good and a bad file: the good one goes ahead, the bad one is reported back
    r = c.post("/api/texts", data={"action": "add", "ocr": "off", "ocr_lang": "eng+hye"},
               files=[("files", ("../../good.pdf", good, "application/pdf")),
                      ("files", ("empty.pdf", b"", "application/pdf"))])
    assert r.status_code == 200, r.text
    assert [s["file"] for s in r.json()["skipped"]] == ["empty.pdf"]
    (cmd,), kw = launched[0][0][2:3], launched[0][1]
    assert "--ocr" in cmd and cmd[cmd.index("--ocr") + 1] == "off" and cmd[cmd.index("--ocr-lang") + 1] == "eng+hye"
    uploaded = cmd[cmd.index("--files") + 1]
    assert os.path.basename(uploaded).endswith("good.pdf") and ".." not in uploaded
    assert os.path.dirname(os.path.dirname(uploaded)) == os.path.join(app.RUNS, "_uploads")
    for f in kw["cleanup"]:
        os.remove(f)
    assert c.get("/api/texts/ocr").json()["available"] in (True, False)
