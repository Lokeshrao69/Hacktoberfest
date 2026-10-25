import io
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import LANGUAGES, Agent
from .llm import LLM, LLMError, load_dotenv
from .scanner import scan
from .search import BM25
from .segment import segment

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
MAX_DOCS = 200
MAX_CHARS = 120_000

load_dotenv(ROOT / ".env")
app = FastAPI(title="ClauseGuard")
llm = LLM()
DOCS: "OrderedDict[str, dict]" = OrderedDict()  # in-memory LRU of analysed documents


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=50, max_length=MAX_CHARS)


class AskIn(BaseModel):
    doc_id: str
    question: str = Field(min_length=2, max_length=2000)
    history: list = []
    language: str = "English"


@app.get("/api/health")
def health():
    return {"ok": True, "model": llm.model, "base_url": llm.base_url, "configured": llm.configured,
            "region": os.getenv("CONTRACT_REGION", "India"), "languages": LANGUAGES}


@app.get("/api/samples")
def samples():
    return [{"name": p.stem, "title": p.read_text(encoding="utf-8").split("\n", 1)[0].strip()}
            for p in sorted(SAMPLES.glob("*.txt"))]


@app.get("/api/samples/{name}")
def sample(name: str):
    p = SAMPLES / f"{name}.txt"
    if not p.is_file() or p.parent != SAMPLES:
        raise HTTPException(404, "no such sample")
    return {"text": p.read_text(encoding="utf-8")}


@app.post("/api/extract")
async def extract(request: Request):
    """Raw-body upload of a .pdf (application/pdf) or text file; returns plain text."""
    body = await request.body()
    if len(body) > 10_000_000:
        raise HTTPException(413, "file too large")
    if body[:4] == b"%PDF":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(body))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(400, f"could not read PDF: {e}")
    else:
        text = body.decode("utf-8", errors="replace")
    return {"text": text[:MAX_CHARS]}


@app.post("/api/analyze")
def analyze(inp: AnalyzeIn):
    if not llm.configured:
        raise HTTPException(503, "No model configured. Set MODEL_ACCESS_KEY (DigitalOcean) or LLM_BASE_URL.")
    t0 = time.time()
    clauses = segment(inp.text)
    if not clauses:
        raise HTTPException(400, "could not find any clauses")
    result = scan(clauses, llm)
    doc_id = uuid.uuid4().hex[:12]
    DOCS[doc_id] = {"clauses": clauses, "index": BM25(clauses), **result}
    while len(DOCS) > MAX_DOCS:
        DOCS.popitem(last=False)
    return {"doc_id": doc_id, "clauses": clauses, "model": llm.model,
            "seconds": round(time.time() - t0, 1), **result}


@app.post("/api/ask")
def ask(inp: AskIn):
    doc = DOCS.get(inp.doc_id)
    if not doc:
        raise HTTPException(404, "Document expired, please analyse it again.")
    DOCS.move_to_end(inp.doc_id)
    try:
        return Agent(llm, doc).run(inp.question, inp.history, inp.language)
    except LLMError as e:
        raise HTTPException(502, f"Model error: {e}")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
