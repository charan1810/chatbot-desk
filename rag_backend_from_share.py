# rag_backend.py — OpenAI-ready & corporate-safe with relevance gating
# - Defaults to BM25 (no internet, no downloads).
# - Optional FAISS+OpenAI embeddings (set EMB_BACKEND=openai).
# - Optional FAISS+HF embeddings (set EMB_BACKEND=faiss + provide local HF model dir).
# - Loads OPENAI_API_KEY from:
#     Windows:  C:\ProgramData\NOVA-99\.env
#     macOS:    ~/Library/Application Support/NOVA-99/.env
# Endpoints: /health, /config, /config/key, /api/ingest-disk, /api/chat

import os, platform, json, re
from pathlib import Path
from typing import List, Literal, Optional
from langchain.schema import SystemMessage, HumanMessage

# ---- TLS: use OS trust store everywhere (fixes corp SSL MITM) ----
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# LLM (OpenAI chat)
from langchain_openai import ChatOpenAI

# Optional: OpenAI embeddings (no HF download)
try:
    from langchain_openai import OpenAIEmbeddings
    HAS_OAI_EMB = True
except Exception:
    HAS_OAI_EMB = False

# ---------- App paths ----------
APP_NAME = "NOVA-99"
IS_WIN = platform.system() == "Windows"
CONFIG_DIR = (Path(os.getenv("PROGRAMDATA", r"C:\ProgramData")) / APP_NAME) if IS_WIN \
             else (Path.home() / "Library" / "Application Support" / APP_NAME)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = CONFIG_DIR / ".env"
SOURCE_FOLDER = CONFIG_DIR / "policies_kb"; SOURCE_FOLDER.mkdir(parents=True, exist_ok=True)
INDEX_DIR = CONFIG_DIR / "index"; INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load env once ----------
def _load_env_once():
    local_env = Path.cwd() / ".env"
    for p in (local_env, ENV_PATH):
        if p.exists():
            load_dotenv(p, override=True)
            break
_load_env_once()

# ---------- Retrieval engine selection ----------
# Options:
#   "bm25"  -> offline keyword retriever (default; zero downloads)
#   "openai"-> FAISS + OpenAI embeddings (paid API; no HF downloads)
#   "faiss" -> FAISS + HuggingFace embeddings (requires model; avoid online by setting HF_LOCAL_MODEL_DIR)
EMB_BACKEND = os.getenv("EMB_BACKEND", "bm25").strip().lower()

# FAISS
USE_FAISS = False
HuggingFaceEmbeddings = None
FAISS = None
if EMB_BACKEND in ("faiss", "openai"):
    try:
        from langchain_community.vectorstores import FAISS
        USE_FAISS = True
    except Exception:
        USE_FAISS = False

# HF embeddings are optional (only if EMB_BACKEND=faiss)
if EMB_BACKEND == "faiss":
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

# BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# ---------- Config ----------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 4
TEMPERATURE = 0.0
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMB_MODEL  = "text-embedding-3-small"   # 1536-dim
HF_MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
HF_LOCAL_MODEL_DIR = os.getenv("HF_LOCAL_MODEL_DIR", "").strip()

# Globals
_vectordb = None       # FAISS object
_bm25 = None           # BM25Okapi object
_chunks: List[str] = []
_embeddings = None

# ---------- Utilities ----------
def _list_pdfs() -> List[Path]:
    return sorted(SOURCE_FOLDER.glob("*.pdf"))

def _load_pdfs(paths: List[Path]):
    docs = []
    for p in paths:
        try:
            if p.exists() and p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")
    return docs

def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def _ensure_embeddings():
    """Return an embeddings object (or None) based on EMB_BACKEND."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if EMB_BACKEND == "openai":
        if not HAS_OAI_EMB:
            print("[warn] OpenAIEmbeddings not available; falling back to BM25")
            return None
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            print("[warn] OPENAI_API_KEY not set; falling back to BM25")
            return None
        _embeddings = OpenAIEmbeddings(model=OPENAI_EMB_MODEL)
        return _embeddings

    if EMB_BACKEND == "faiss" and HuggingFaceEmbeddings is not None:
        if HF_LOCAL_MODEL_DIR and Path(HF_LOCAL_MODEL_DIR).exists():
            _embeddings = HuggingFaceEmbeddings(model_name=HF_LOCAL_MODEL_DIR)
        else:
            _embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
        return _embeddings

    return None

def _build_faiss():
    """Build/load FAISS index with whichever embeddings are configured."""
    global _vectordb
    if not USE_FAISS:
        return None

    emb = _ensure_embeddings()
    if emb is None:
        return None

    faiss_bin = INDEX_DIR / "index.faiss"
    faiss_meta = INDEX_DIR / "index.pkl"
    if faiss_bin.exists() and faiss_meta.exists():
        _vectordb = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
        print(f"[info] Loaded FAISS index from {INDEX_DIR}")
        return _vectordb

    pdfs = _list_pdfs()
    if not pdfs:
        print(f"[warn] No PDFs at {SOURCE_FOLDER}")
        return None
    raw = _load_pdfs(pdfs)
    if not raw:
        print("[warn] 0 pages loaded"); return None
    chunks = _split_docs(raw)
    if not chunks:
        print("[warn] 0 chunks produced"); return None

    _vectordb = FAISS.from_documents(chunks, emb)
    FAISS.save_local(_vectordb, INDEX_DIR)
    print(f"[info] Saved FAISS index to {INDEX_DIR}")
    return _vectordb

def _build_bm25():
    """Pure-Python fallback: no embeddings, no internet."""
    global _bm25, _chunks
    (INDEX_DIR / "bm25").mkdir(exist_ok=True)
    meta = INDEX_DIR / "bm25" / "meta.json"
    store = INDEX_DIR / "bm25" / "chunks.json"

    # load cache
    if meta.exists() and store.exists():
        try:
            _chunks = json.loads(store.read_text(encoding="utf-8"))
            corpus = [c.split() for c in _chunks]
            _bm25 = BM25Okapi(corpus)
            print(f"[info] Loaded BM25 index with {len(_chunks)} chunks")
            return True
        except Exception as e:
            print(f"[warn] Failed to load BM25 cache: {e}")

    pdfs = _list_pdfs()
    if not pdfs:
        print(f"[warn] No PDFs at {SOURCE_FOLDER}"); return False
    raw = _load_pdfs(pdfs)
    if not raw:
        print("[warn] 0 pages loaded"); return False
    chunks = _split_docs(raw)
    if not chunks:
        print("[warn] 0 chunks produced"); return False

    _chunks = [d.page_content for d in chunks]
    corpus = [c.split() for c in _chunks]
    _bm25 = BM25Okapi(corpus)

    meta.write_text(json.dumps({"count": len(_chunks)}), encoding="utf-8")
    store.write_text(json.dumps(_chunks), encoding="utf-8")
    print(f"[info] Built BM25 index with {len(_chunks)} chunks")
    return True

def _retrieve_texts(query: str, k: int = TOP_K) -> List[str]:
    if USE_FAISS and _vectordb is not None:
        docs = _vectordb.similarity_search(query, k=k)
        return [d.page_content for d in docs]
    if HAS_BM25 and _bm25 is not None and _chunks:
        scores = _bm25.get_scores(query.split())
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [_chunks[i] for i in idx]
    return []

# ---- Relevance gate: treat weak matches as "no context" so LLM can small-talk/help ----
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")
def _keyword_overlap(q: str, t: str) -> int:
    qs = set(_WORD_RE.findall(q.lower()))
    ts = set(_WORD_RE.findall(t.lower()))
    return len(qs & ts)

def _looks_relevant(q: str, ctx: List[str]) -> bool:
    """Return True if any chunk shares at least 2 content-words with the query."""
    if not ctx:
        return False
    best = max((_keyword_overlap(q, c) for c in ctx), default=0)
    return best >= 2

def _ensure_index():
    """Try FAISS (if selected), else BM25. Never raises."""
    global _vectordb
    if EMB_BACKEND in ("openai", "faiss") and USE_FAISS:
        try:
            if _vectordb is None:
                _build_faiss()
            if _vectordb is not None:
                return "faiss"
        except Exception as e:
            print(f"[warn] FAISS failed, falling back to BM25: {e}")

    if HAS_BM25:
        ok = _build_bm25()
        return "bm25" if ok else "none"

    return "none"

def _llm() -> Optional[ChatOpenAI]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    # If you use a gateway/proxy, set OPENAI_BASE_URL env.
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=TEMPERATURE, timeout=60)

# ---------- FastAPI ----------
app = FastAPI(title="RAG Desktop Backend (OpenAI/BM25 corp-safe)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Msg(BaseModel):
    role: Literal["user","assistant","system"]
    text: str

class ChatRequest(BaseModel):
    messages: List[Msg]
    top_k: int | None = None

class KeyPayload(BaseModel):
    key: str

@app.get("/health")
def health():
    engine = _ensure_index()
    key_set = bool(os.getenv("OPENAI_API_KEY", "").strip())
    pdfs = len(_list_pdfs())
    has_index = (INDEX_DIR / "index.faiss").exists() or (INDEX_DIR / "bm25" / "chunks.json").exists()
    return {
        "ok": True,
        "engine": engine,
        "emb_backend": EMB_BACKEND,
        "key": key_set,
        "pdfs": pdfs,
        "has_index": has_index,
        "source_folder": str(SOURCE_FOLDER),
        "index_dir": str(INDEX_DIR),
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Professional replies in ALL cases:
    - If relevant RAG context exists -> strict RAG answer only from context.
    - If context is weak/unrelated or empty -> general, helpful reply (small talk, clarifications),
      but never invent internal policy details.
    """
    q = next((m.text.strip() for m in reversed(req.messages) if m.role == "user"), "")
    if not q:
        return {"text": "Please ask a question."}

    # Make sure retrieval engine is ready
    engine = _ensure_index()
    llm = _llm()

    # If no index at all, let LLM handle general small-talk/help
    if engine == "none":
        if llm is None:
            return {"text": "(No API key set) I’m online but can’t reach a model right now."}
        msgs = [
            SystemMessage(content=(
                "You are NOVA-99, a professional and friendly company assistant. "
                "If the user asks about company policies or documents and you do not have any context loaded, "
                "explain that you can’t access those documents yet and suggest adding PDFs and clicking Re-index. "
                "For greetings or general chit-chat, respond politely and briefly. "
                "Do not invent policy details. Keep answers concise and professional."
            )),
            HumanMessage(content=q),
        ]
        ans = llm.invoke(msgs)
        txt = ans if isinstance(ans, str) else getattr(ans, "content", str(ans))
        return {"text": txt.strip()}

    # With an engine, try to retrieve (k can be 0)
    k = max(1, int(req.top_k or TOP_K))
    ctx = _retrieve_texts(q, k=k) if k > 0 else []

    # Relevance gating: if ctx exists but looks unrelated, drop it so LLM can small-talk/help
    if ctx and not _looks_relevant(q, ctx):
        ctx = []

    if llm is None:
        # No key; still be helpful
        if ctx:
            preview = "\n\n---\n\n".join(ctx[:3])
            return {"text": f"(No API key set — answering with retrieved text only)\n\n{preview}"}
        return {"text": "(No API key set) I’m online but can’t reach a model right now."}

    try:
        if ctx:
            # STRICT RAG when we have relevant context
            prompt = (
                "You are a professional assistant. Answer from the provided context. "
                "If the answer is not in the context, say Sorry that is not in the context of this conversation.\n\n"
                f"Question: {q}\n\nContext:\n" + "\n\n---\n\n".join(ctx[:max(1, k)])
            )
            ans = llm.invoke(prompt)
            # print("hello1hello")
            txt = ans if isinstance(ans, str) else getattr(ans, "content", str(ans))
            # print("hello1")
            return {"text": txt.strip()}
        else:
            # NO (relevant) CONTEXT: general, bounded answer
            msgs = [
                SystemMessage(content=(
                    "You are NOVA-99, a professional and friendly company assistant. "
                    "For greetings or general questions, respond politely and helpfully. "
                    "If the user asks about company policies or internal documents and you lack context, "
                    "do NOT invent details—explain that the documents aren’t loaded yet and suggest adding PDFs and clicking Re-index. "
                    "Keep responses concise and professional."
                )),
                HumanMessage(content=q),
            ]
            # print("hello2")
            ans = llm.invoke(msgs)
            txt = ans if isinstance(ans, str) else getattr(ans, "content", str(ans))
            # print("hello3")
            return {"text": txt.strip()}
    except Exception as e:
        return {"text": "Error connecting to Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000, reload=False)
