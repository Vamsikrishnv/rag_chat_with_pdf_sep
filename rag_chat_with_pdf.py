# rag_chat_with_pdf.py — multi-PDF, named indexes, incremental updates
import os
import sys
import argparse
import pathlib
import json
from typing import List, Optional, Iterable, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# -----------------------------
# Providers
# -----------------------------
def get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Edit your .env.")
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    elif provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        model = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider}")


def get_llm():
    from langchain_openai import ChatOpenAI
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


# -----------------------------
# Splitters
# -----------------------------
def token_text_splitter(chunk_size=350, chunk_overlap=50):
    try:
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter
        return SentenceTransformersTokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )


def build_text_splitter(strategy: str, chunk_size: int, overlap: int):
    s = strategy.lower()
    if s == "fixed":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    if s == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
    if s == "token":
        return token_text_splitter(chunk_size=chunk_size, chunk_overlap=overlap)
    raise ValueError("strategy must be: fixed | recursive | token")


# -----------------------------
# Storage helpers
# -----------------------------
def ensure_storage_dir(strategy: str, index_name: Optional[str]) -> str:
    tag = index_name.strip().lower().replace(" ", "_") if index_name else strategy
    d = os.path.join("storage", f"faiss_{strategy}_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def ids_sidecar_path(store_dir: str) -> str:
    return os.path.join(store_dir, "ids.json")


def load_seen_ids(store_dir: str) -> set:
    p = ids_sidecar_path(store_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_seen_ids(store_dir: str, seen: Iterable[str]) -> None:
    p = ids_sidecar_path(store_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen)), f)


def maybe_load_faiss(store_dir: str, embeddings):
    try:
        return FAISS.load_local(
            store_dir, embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception:
        return None


# -----------------------------
# PDF discovery & loading
# -----------------------------
def discover_pdfs(docs_dir: Optional[str]) -> List[pathlib.Path]:
    if not docs_dir:
        return []
    root = pathlib.Path(docs_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"--docs not found: {root}")
    files = [p for p in root.rglob("*.pdf")]
    return files


def normalize_path(p: str) -> pathlib.Path:
    return pathlib.Path(p).resolve()


def load_pdf_docs(pdf_path: pathlib.Path) -> List[Document]:
    # PyPDFLoader prefers POSIX slashes even on Windows
    clean_posix = pdf_path.as_posix()
    loader = PyPDFLoader(clean_posix)
    docs = loader.load()
    out = []
    for idx, d in enumerate(docs, start=1):
        meta = dict(d.metadata)
        # filename for source in citations
        meta["source"] = pdf_path.name
        # robust page fallback: prefer PDF metadata; else use enumeration (1-based)
        page = meta.get("page") or meta.get("page_number") or idx
        try:
            page = int(page)
        except Exception:
            page = idx
        meta["page"] = page
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out



def chunk_docs(splitter, docs: List[Document]) -> List[Document]:
    return splitter.split_documents(docs)


def materialize(
    splitter,
    pdfs: List[pathlib.Path],
    store_dir: str,
    embeddings,
    reindex: bool
) -> Tuple[FAISS, int, int]:
    """
    Build or update the FAISS index:
    - If existing and not --reindex, incrementally add unseen chunks (based on ids.json).
    - If --reindex or no index, rebuild from scratch.
    Returns: (vectorstore, num_pdfs_processed, num_chunks_added)
    """
    console = Console()
    seen_ids = set()
    vs = None

    if not reindex:
        vs = maybe_load_faiss(store_dir, embeddings)
        seen_ids = load_seen_ids(store_dir)

    # When reindexing, reset
    if reindex or vs is None:
        seen_ids = set()
        vs = None
        # clean previous sidecar on full rebuild
        try:
            os.remove(ids_sidecar_path(store_dir))
        except Exception:
            pass

    all_new_chunks = []
    pdf_count = 0
    total_chunks = 0

    for pdf in pdfs:
        if not pdf.exists():
            console.print(f"[yellow]Skip missing PDF:[/yellow] {pdf}")
            continue

        console.print(f"[bold]Loading PDF[/bold]: {pdf.as_posix()}")
        raw_docs = load_pdf_docs(pdf)
        chunks = chunk_docs(splitter, raw_docs)

        new_for_this_pdf = []
        for i, ch in enumerate(chunks):
            # make a stable chunk id: filename|page|index
            cid = f"{ch.metadata.get('source')}|{ch.metadata.get('page')}|{i}"
            if cid in seen_ids:
                continue
            # keep id in metadata for debugging
            ch.metadata["chunk_id"] = cid
            new_for_this_pdf.append(ch)
            seen_ids.add(cid)

        console.print(f"[green]Chunks:[/green] {len(chunks)} | New: {len(new_for_this_pdf)}")
        all_new_chunks.extend(new_for_this_pdf)
        pdf_count += 1
        total_chunks += len(new_for_this_pdf)

    console.print(f"[bold cyan]Total new chunks to add:[/bold cyan] {total_chunks}")

    if vs is None:
        if not all_new_chunks:
            raise RuntimeError("Nothing to index. Provide PDFs or use --reindex.")
        vs = FAISS.from_documents(all_new_chunks, embeddings)
    else:
        if all_new_chunks:
            vs.add_documents(all_new_chunks)

    # Persist index + sidecar ids
    vs.save_local(store_dir)
    save_seen_ids(store_dir, seen_ids)
    return vs, pdf_count, total_chunks


# -----------------------------
# Reranking (optional)
# -----------------------------
def rerank_with_cross_encoder(query: str, docs: List[Document]):
    try:
        from sentence_transformers import CrossEncoder
        model_id = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoder(model_id, device="cpu")
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs).tolist()
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        ds, sc = zip(*ranked)
        return list(ds), list(sc)
    except Exception:
        return docs, None


# -----------------------------
# Prompt Context
# -----------------------------
def build_context(docs: List[Document], max_chars=3500) -> str:
    parts, total = [], 0
    for d in docs:
        page = d.metadata.get("page", "?")
        src = d.metadata.get("source", "doc")
        block = f"[{src} p.{page}] " + d.page_content
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    console = Console()

    ap = argparse.ArgumentParser(description="Chat with PDFs via LangChain RAG (FAISS).")
    # Ingestion
    ap.add_argument("--pdf", action="append", help="Path to a PDF (can repeat).")
    ap.add_argument("--docs", type=str, help="Directory of PDFs (recurses).")
    ap.add_argument("--index-name", type=str, default=None, help="Name this corpus (e.g., kb, resume, research).")
    ap.add_argument("--reindex", action="store_true", help="Force rebuild of the index for this corpus.")
    # Retrieval & chunking
    ap.add_argument("--strategy", type=str, default="recursive", help="fixed|recursive|token")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mmr", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    try:
        embeddings = get_embeddings()
        llm = get_llm()
    except Exception as e:
        console.print(f"[red]Provider init error:[/red] {e}")
        sys.exit(1)

    splitter = build_text_splitter(args.strategy, args.chunk_size, args.overlap)
    store_dir = ensure_storage_dir(args.strategy, args.index_name)

    # Build the list of PDFs to process (for update or build)
    pdfs: List[pathlib.Path] = []
    if args.pdf:
        pdfs.extend([normalize_path(p) for p in args.pdf])
    if args.docs:
        pdfs.extend(discover_pdfs(args.docs))
    # If no PDFs are provided, we will attempt to load an existing index

    # Build/update index
    vs = maybe_load_faiss(store_dir, embeddings)
    if args.reindex or vs is None or pdfs:
        try:
            vs, pdf_count, new_chunks = materialize(
                splitter=splitter,
                pdfs=pdfs,
                store_dir=store_dir,
                embeddings=embeddings,
                reindex=args.reindex or (vs is None and not pdfs)
            )
            console.print(f"[green]Index ready[/green] at {store_dir} | PDFs processed: {pdf_count} | New chunks: {new_chunks}")
        except Exception as e:
            console.print(f"[red]Indexing error:[/red] {e}")
            if vs is None:
                sys.exit(1)
    else:
        console.print(f"[green]Loaded existing FAISS index from[/green] {store_dir}")

    # Build retriever
    if args.mmr:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": args.k, "fetch_k": max(args.k * 4, 20), "lambda_mult": 0.5},
        )
    else:
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": args.k})

    # Prompt template
    try:
        from prompts import get_rag_prompt
        prompt = get_rag_prompt()
    except Exception:
        from langchain.prompts import ChatPromptTemplate
        system = (
            "You are a precise research assistant. Answer using only the provided context. "
            "If the answer is not in the context, say you don't know. Cite sources as (filename p.N)."
        )
        human = "Question:\n{question}\n\nContext:\n{context}\n\nReturn a concise answer with citations."
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    console.print("\n[bold cyan]RAG chat ready.[/bold cyan] Type your question (or 'exit').\n")
    while True:
        q = Prompt.ask("[bold yellow]Q[/bold yellow]")
        if q.strip().lower() in {"exit", "quit"}:
            break

        try:
            docs = retriever.get_relevant_documents(q)
        except Exception as e:
            console.print(f"[red]Retrieval error:[/red] {e}")
            continue

        scores = None
        if args.rerank:
            docs, scores = rerank_with_cross_encoder(q, docs)

        # Optional: pretty source table if you have utils.py with print_sources
        try:
            from utils import print_sources
            print_sources(docs, scores=scores)
        except Exception:
            # Minimal inline sources
            console.print("[dim]Sources:[/dim] " + ", ".join(
                f"{d.metadata.get('source','doc')} p.{d.metadata.get('page','?')}" for d in docs
            ))

        ctx = build_context(docs, max_chars=5000)
        chain = prompt | llm
        try:
            ans = chain.invoke({"question": q, "context": ctx}).content
        except Exception as e:
            console.print(f"[red]LLM error:[/red] {e}")
            continue

        console.print("\n[bold green]Answer[/bold green]:")
        console.print(ans)
        console.print("")


if __name__ == "__main__":
    main()
