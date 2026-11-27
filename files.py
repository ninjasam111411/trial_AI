"""
Files you attach with ➕. Neural reads them, splits them into passages, and
the Files agent answers questions by finding the passage that fits best.

Reads: .txt .md .csv .json and code files, .pdf, .docx
"""

import json
import re
import time
import uuid

from config import SAVE_DIR

FILES_DIR = SAVE_DIR / "files"
TEXT_TYPES = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".css", ".lua",
              ".luau", ".xml", ".log", ".yml", ".yaml", ".ini", ".cfg", ".java", ".c", ".cpp",
              ".cs", ".rb", ".go", ".rs", ".sh", ".bat", ".sql", ".rtf"}
MAX_BYTES = 20 * 1024 * 1024
STOP = set("""a an the is are was were be been to of in on at for with about and or but so if it its
this that these those what which who whom how why when where do does did can could would should will
i me my you your we our they their he she his her them there here as by from not no yes just very
tell explain give show please file document doc says say said""".split())


class FileError(Exception):
    pass


def extract_text(name, data):
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if len(data) > MAX_BYTES:
        raise FileError("That file is over 20 MB. Try a smaller one.")
    if ext in TEXT_TYPES:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from io import BytesIO

            from pypdf import PdfReader
        except ImportError:
            raise FileError("Reading PDFs needs the pypdf package (pip install pypdf).")
        reader = PdfReader(BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        try:
            from io import BytesIO

            import docx
        except ImportError:
            raise FileError("Reading Word files needs python-docx (pip install python-docx).")
        d = docx.Document(BytesIO(data))
        return "\n\n".join(p.text for p in d.paragraphs)
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        raise FileError("I can't see pictures yet. I can read text, PDF, Word and code files.")
    raise FileError(f"I can't read {ext or 'that kind of'} files yet. "
                    "Try text, PDF, Word (.docx) or code files.")


def chunk(text, size=500):
    """One passage per paragraph (tiny ones join the next; long ones split
    on sentences), so answers quote just the part that matters."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r", "")) if p.strip()]
    if len(paras) <= 1:  # no blank lines (common in PDFs): use single lines
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, carry = [], ""
    for p in paras:
        p = (carry + " " + p).strip() if carry else p
        carry = ""
        if len(p) < 60:
            carry = p
            continue
        while len(p) > size:
            cut = max(p.rfind(". ", 0, size), p.rfind("? ", 0, size), p.rfind("! ", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    if carry:
        chunks.append(carry)
    return chunks


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def keywords(text):
    return {_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def save_file(owner, name, data):
    text = extract_text(name, data)
    if not text.strip():
        raise FileError("I couldn't find any text in that file (scanned PDFs are pictures of text).")
    doc = {"id": uuid.uuid4().hex[:12], "owner": owner, "name": name, "time": int(time.time()),
           "words": len(text.split()), "chunks": chunk(text)}
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    (FILES_DIR / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def load_file(file_id, owner):
    try:
        doc = json.loads((FILES_DIR / f"{file_id}.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return doc if doc.get("owner") == owner else None


def search(docs, question):
    """Best (score, doc name, passage) across the attached files. Score is the
    share of the question's important words found in the passage (0-1)."""
    q = keywords(question)
    if not q:
        return 0.0, None, None
    best = (0.0, None, None)
    for doc in docs:
        for passage in doc["chunks"]:
            hit = len(q & keywords(passage)) / len(q)
            if hit > best[0]:
                best = (hit, doc["name"], passage)
    return best
