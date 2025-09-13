# streamlit_app.py — Chat with PDFs (LangChain + FAISS) with a simple UI
# Features: file upload, named corpora, incremental indexing, similarity/MMR, hybrid (BM25+embeddings),
# rerank, sources table, delete index, refresh index info, perf metrics, export chat CSV.

import os
import json
import time
import shutil
import pathlib
import io
import csv
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# -----------------------------
# Providers
# -----------------------------
def get_embeddings(provider: str, openai_model: str, hf_model: str):
    """Return an embeddings object based on provider selection."""
    provider = (provider or os.getenv("EMBEDDINGS_PROVIDER", "openai")).lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = openai_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or Streamlit secrets.")
        return OpenAIEmbeddings(model=model)
    elif provider in {"hf", "huggingface"}:
        from langchain_huggingface import HuggingFaceEmbeddings
        model = hf_model or os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def get_llm():
    """LLM used for generating the final answer (with context)."""
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
            separators=["\n\n", "\n", " ", ""]
        )


def build_text_splitter(strategy: str, chunk_size: int, overlap: int):
    s = (strategy or "recursive").lower()
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
# Storage helpers (FAISS)
# -----------------------------
def ensure_storage_dir(strategy: str, index_name: Optional[str]) -> str:
    """Create/return the folder where this index lives."""
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


def save_seen_ids(store_dir: str, seen) -> None:
    p = ids_sidecar_path(store_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen)), f)


def maybe_load_faiss(store_dir: str, embeddings):
    try:
        return FAISS.load_local(store_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


# -----------------------------
# Index discovery & utilities
# -----------------------------
def list_existing_corpora() -> List[str]:
    """List existing FAISS index names (based on storage folder names)."""
    root = pathlib.Path("storage")
    if not root.exists():
        return []
    names = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("faiss_"):
            parts = d.name.split("_", 2)  # faiss_strategy_indexname
            if len(parts) >= 3:
                names.append(parts[-1])
    return sorted(set(names))


def delete_index_dir(store_dir: str):
    """Delete the entire FAISS index folder."""
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)


def peek_index_sources(store_dir: str, embeddings) -> list[str]:
    """Return a quick list of source filenames in an index (first ~1000 docs)."""
    try:
        vs = FAISS.load_local(store_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return []
    try:
        ids = list(vs.index_to_docstore_id.values())
        srcs = set()
        for i in ids[:1000]:
            d = vs.docstore.search(i)
            if d and isinstance(d, Document):
                srcs.add(d.metadata.get("source", "doc"))
        return sorted(srcs)
    except Exception:
        return []


# -----------------------------
# PDF loading & chunking
# -----------------------------
def load_pdf_docs(pdf_path: pathlib.Path) -> List[Document]:
    """Load a PDF into per-page Documents with robust page/source metadata."""
    clean_posix = pdf_path.as_posix()
    loader = PyPDFLoader(clean_posix)
    docs = loader.load()
    out = []
    for idx, d in enumerate(docs, start=1):
        meta = dict(d.metadata)
        meta["source"] = pdf_path.name
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


def materialize(splitter, pdfs: List[pathlib.Path], store_dir: str, embeddings, reindex: bool):
    """
    Build or update the FAISS index:
    - If existing and not reindex, incrementally add unseen chunks (ids.json).
    - If reindex or no index, rebuild from scratch.
    Returns: (vectorstore, num_pdfs_processed, num_chunks_added)
    """
    seen_ids = set()
    vs = None

    if not reindex:
        vs = maybe_load_faiss(store_dir, embeddings)
        seen_ids = load_seen_ids(store_dir)

    if reindex or vs is None:
        seen_ids = set()
        vs = None
        try:
            os.remove(ids_sidecar_path(store_dir))
        except Exception:
            pass

    all_new_chunks = []
    pdf_count = 0
    total_chunks = 0

    for pdf in pdfs:
        if not pdf.exists():
            st.warning(f"Skip missing PDF: {pdf}")
            continue

        st.write(f"**Loading PDF:** {pdf.as_posix()}")
        raw_docs = load_pdf_docs(pdf)
        chunks = chunk_docs(splitter, raw_docs)

        new_for_this_pdf = []
        for i, ch in enumerate(chunks):
            cid = f"{ch.metadata.get('source')}|{ch.metadata.get('page')}|{i}"
            if cid in seen_ids:
                continue
            ch.metadata["chunk_id"] = cid
            new_for_this_pdf.append(ch)
            seen_ids.add(cid)

        st.write(f"Chunks: {len(chunks)} | New: {len(new_for_this_pdf)}")
        all_new_chunks.extend(new_for_this_pdf)
        pdf_count += 1
        total_chunks += len(new_for_this_pdf)

    st.info(f"Total new chunks to add: {total_chunks}")

    if vs is None:
        if not all_new_chunks:
            raise RuntimeError("Nothing to index. Provide PDFs or toggle Reindex.")
        vs = FAISS.from_documents(all_new_chunks, embeddings)
    else:
        if all_new_chunks:
            vs.add_documents(all_new_chunks)

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
    except Exception as e:
        st.warning(f"Rerank disabled (missing model or package): {e}")
        return docs, None


# -----------------------------
# Build context for LLM
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
# Hybrid helpers (BM25 + RRF)
# -----------------------------
def load_all_docs_from_index(vs) -> List[Document]:
    try:
        ids = list(vs.index_to_docstore_id.values())
    except Exception:
        return []
    docs = []
    for i in ids[:2000]:
        d = vs.docstore.search(i)
        if d and isinstance(d, Document):
            docs.append(d)
    return docs


def rrf_fuse(result_lists: list[list[Document]], k: int, k_base: int = 60) -> list[Document]:
    scores = {}
    order = []

    def key_of(d: Document):
        return (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chunk_id"),
            id(d),
        )

    for results in result_lists:
        for rank, d in enumerate(results, start=1):
            kdoc = key_of(d)
            if kdoc not in scores:
                scores[kdoc] = 0.0
                order.append(d)
            scores[kdoc] += 1.0 / (k_base + rank)

    sorted_docs = sorted(order, key=lambda d: scores[key_of(d)], reverse=True)
    return sorted_docs[:k]


def get_or_build_bm25(vs, store_dir: str, bm25_k: int):
    key = f"bm25::{store_dir}"
    if "bm25_cache" not in st.session_state:
        st.session_state.bm25_cache = {}
    entry = st.session_state.bm25_cache.get(key)
    if entry:
        bm25 = entry
    else:
        all_docs = load_all_docs_from_index(vs)
        bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
        st.session_state.bm25_cache[key] = bm25
    if bm25:
        bm25.k = int(bm25_k)
    return bm25


# -----------------------------
# Chat export helper
# -----------------------------
def chat_to_csv(messages: list[tuple[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["role", "message"])
    for role, msg in messages:
        w.writerow([role, msg.replace("\n", " ")])
    return buf.getvalue().encode("utf-8")


def run_once(question: str,
             vs,
             store_dir: str,
             k: int,
             use_mmr: bool,
             lambda_mult: float,
             use_hybrid: bool,
             bm25_k: int,
             use_rerank: bool):
    if use_mmr:
        dense_retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": int(k), "fetch_k": max(int(k)*4, 20), "lambda_mult": float(lambda_mult)},
        )
    else:
        dense_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})

    t0 = time.time()

    if use_hybrid:
        bm25 = get_or_build_bm25(vs, store_dir, bm25_k)
        sparse_docs = bm25.get_relevant_documents(question) if bm25 else []
        dense_docs = dense_retriever.invoke(question)
        docs = rrf_fuse([dense_docs, sparse_docs], k=int(k))
    else:
        docs = dense_retriever.invoke(question)

    t_retr = time.time()

    scores = None
    if use_rerank:
        docs, scores = rerank_with_cross_encoder(question, docs)
    t_rerank = time.time()

    ctx = build_context(docs, max_chars=5000)

    from langchain.prompts import ChatPromptTemplate
    system = (
        "You are a precise research assistant. Answer using only the provided context. "
        "If the answer is not in the context, say you don't know. Cite sources as (filename p.N)."
    )
    human = "Question:\n{question}\n\nContext:\n{context}\n\nReturn a concise answer with citations."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    llm = get_llm()
    ans = (prompt | llm).invoke({"question": question, "context": ctx}).content
    t_llm = time.time()

    timings = {
        "t_retrieval": t_retr - t0,
        "t_rerank": t_rerank - t_retr,
        "t_llm": t_llm - t_rerank
    }
    return ans, docs, timings


# -----------------------------
# UI
# -----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs (RAG)", page_icon="📘", layout="wide")
    st.title("📘 Chat with PDFs — RAG (LangChain + FAISS)")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        refresh_bm25 = st.button("Refresh BM25 cache")

        existing = list_existing_corpora()
        index_name = st.selectbox(
            "Index name",
            options=(["kb"] + [n for n in existing if n != "kb"]),
            index=0,
            help="Keeps corpora separate (stored under storage/)"
        )

        strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "token"], index=0)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=3000, value=1200, step=50)
        overlap = st.number_input("Overlap", min_value=0, max_value=500, value=200, step=10)

        k = st.slider("k (top results)", 1, 12, 6)

        use_mmr = st.checkbox("Use MMR (diverse retrieval)", value=True)
        lambda_mult = 0.5
        if use_mmr:
            lambda_mult = st.slider("MMR λ (diversity vs relevance)", 0.0, 1.0, 0.5, 0.05)

        st.subheader("Hybrid")
        use_hybrid = st.checkbox("Use Hybrid (BM25 + Embeddings)", value=False)
        bm25_k = st.slider("BM25 k (sparse top results)", 1, 20, 8)

        use_rerank = st.checkbox("Use Cross-Encoder Rerank", value=True)

        reindex = st.checkbox("Reindex from scratch", value=False)

        st.divider()
        provider = st.selectbox("Embeddings provider", ["openai", "hf"], index=0)
        openai_embed_model = st.text_input("OpenAI embed model", value="text-embedding-3-small")
        hf_model = st.text_input("HF model (embeddings)", value="sentence-transformers/all-MiniLM-L6-v2")

        st.divider()
        uploaded_files = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True)
        ingest_btn = st.button("Ingest / Update Index")

        danger = st.checkbox("Enable dangerous actions", value=False)
        if danger and st.button("❗ Delete this index folder"):
            try:
                store_dir = ensure_storage_dir(strategy, index_name)
                delete_index_dir(store_dir)
                st.success(f"Deleted: {store_dir}")
            except Exception as e:
                st.error(f"Delete failed: {e}")

        if st.button("Refresh index info"):
            try:
                store_dir = ensure_storage_dir(strategy, index_name)
                embeddings = get_embeddings(provider, openai_embed_model, hf_model)
                files = peek_index_sources(store_dir, embeddings)
                if files:
                    st.success(f"Files in '{index_name}':\n- " + "\n- ".join(files))
                else:
                    st.info("No files found yet. Upload PDFs and click Ingest.")
            except Exception as e:
                st.error(f"Peek failed: {e}")

    store_dir = ensure_storage_dir(strategy, index_name)

    if refresh_bm25:
        if "bm25_cache" in st.session_state:
            for k2 in list(st.session_state.bm25_cache.keys()):
                if k2.startswith("bm25::"):
                    del st.session_state.bm25_cache[k2]
            st.success("BM25 cache cleared. It will rebuild on next question.")
        else:
            st.info("No BM25 cache yet.")

    status = st.empty()
    if ingest_btn:
        try:
            saved_paths: List[pathlib.Path] = []
            if uploaded_files:
                updir = pathlib.Path("uploads") / index_name
                updir.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    dest = updir / f.name
                    with open(dest, "wb") as out:
                        out.write(f.read())
                    saved_paths.append(dest.resolve())
                st.success(f"Saved {len(saved_paths)} file(s) to {updir}")

            embeddings = get_embeddings(provider, openai_embed_model, hf_model)
            splitter = build_text_splitter(strategy, chunk_size, overlap)

            vs, pdf_count, new_chunks = materialize(
                splitter=splitter,
                pdfs=saved_paths,
                store_dir=store_dir,
                embeddings=embeddings,
                reindex=reindex,
            )
            status.success(f"Index ready at {store_dir} | PDFs processed: {pdf_count} | New chunks: {new_chunks}")
        except Exception as e:
            status.error(f"Indexing error: {e}")

    st.divider()
    tab_chat, tab_eval = st.tabs(["💬 Chat", "🧪 Evaluate"])

    # ---------------- Chat Tab ----------------
    with tab_chat:
        st.subheader("Chat")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        question = st.text_input("Ask a question about your PDFs")
        ask = st.button("Ask")

        if ask and question.strip():
            try:
                embeddings = get_embeddings(provider, openai_embed_model, hf_model)
                vs = maybe_load_faiss(store_dir, embeddings)
                if vs is None:
                    st.warning("No index found yet. Upload PDFs and click 'Ingest / Update Index'.")
                else:
                    if use_mmr:
                        dense_retriever = vs.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": int(k), "fetch_k": max(int(k) * 4, 20), "lambda_mult": float(lambda_mult)},
                        )
                    else:
                        dense_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})

                    t0 = time.time()

                    if use_hybrid:
                        bm25 = get_or_build_bm25(vs, store_dir, bm25_k)
                        sparse_docs = bm25.get_relevant_documents(question) if bm25 else []
                        dense_docs = dense_retriever.invoke(question)
                        docs = rrf_fuse([dense_docs, sparse_docs], k=int(k))
                    else:
                        docs = dense_retriever.invoke(question)

                    t_retrieve = time.time()

                    scores = None
                    if use_rerank:
                        docs, scores = rerank_with_cross_encoder(question, docs)
                    t_rerank = time.time()

                    ctx = build_context(docs, max_chars=5000)

                    from langchain.prompts import ChatPromptTemplate
                    system = (
                        "You are a precise research assistant. Answer using only the provided context. "
                        "If the answer is not in the context, say you don't know. Cite sources as (filename p.N)."
                    )
                    human = "Question:\n{question}\n\nContext:\n{context}\n\nReturn a concise answer with citations."
                    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

                    llm = get_llm()
                    chain = prompt | llm
                    ans = chain.invoke({"question": question, "context": ctx}).content
                    t_llm = time.time()

                    st.session_state.messages.append(("user", question))
                    st.session_state.messages.append(("assistant", ans))

                    st.markdown(f"**Assistant:** {ans}")

                    src_rows = []
                    for rank, d in enumerate(docs, start=1):
                        src = d.metadata.get("source", "doc")
                        page = d.metadata.get("page", "?")
                        preview = (d.page_content or "").strip().replace("\n", " ")
                        if len(preview) > 140:
                            preview = preview[:137] + "..."
                        src_rows.append((rank, src, page, preview))

                    with st.expander("Sources"):
                        st.markdown("| Rank | File | Page | Preview |")
                        st.markdown("|---:|---|---:|---|")
                        for r, s, p, pv in src_rows:
                            st.markdown(f"| {r} | {s} | {p} | {pv} |")

                    st.caption(
                        f"⏱ Retrieval: {(t_retrieve - t0):.2f}s | "
                        f"Rerank: {(t_rerank - t_retrieve):.2f}s | "
                        f"LLM: {(t_llm - t_rerank):.2f}s"
                    )

            except Exception as e:
                st.error(f"Chat error: {e}")

        if st.session_state.get("messages"):
            st.markdown("### Conversation")
            for role, msg in st.session_state.messages:
                if role == "user":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**Assistant:** {msg}")

            csv_bytes = chat_to_csv(st.session_state.messages)
            st.download_button(
                "Download chat as CSV",
                data=csv_bytes,
                file_name=f"chat_{index_name}.csv",
                mime="text/csv"
            )

    # ---------------- Eval Tab ----------------
    with tab_eval:
        st.subheader("Evaluate retrieval modes")

        st.markdown("Paste 1 question per line:")
        q_text = st.text_area("Questions", height=180, placeholder="e.g.\nWhat are the vector DBs discussed?\nHow is MMR different from similarity?")
        modes = st.multiselect(
            "Modes to run",
            ["similarity", "mmr", "hybrid", "similarity+rerank", "mmr+rerank", "hybrid+rerank"],
            default=["similarity", "mmr", "hybrid"]
        )
        run_eval = st.button("Run evaluation")

        if run_eval:
            if not q_text.strip():
                st.warning("Please paste at least one question.")
            else:
                try:
                    embeddings = get_embeddings(provider, openai_embed_model, hf_model)
                    vs = maybe_load_faiss(store_dir, embeddings)
                    if vs is None:
                        st.warning("No index found yet. Upload PDFs and click 'Ingest / Update Index'.")
                    else:
                        rows = []
                        questions = [q.strip() for q in q_text.splitlines() if q.strip()]
                        prog = st.progress(0.0)
                        total = max(1, len(questions) * max(1, len(modes)))
                        step = 0

                        for q in questions:
                            for m in modes:
                                use_mmr_m = "mmr" in m
                                use_hybrid_m = "hybrid" in m
                                use_rerank_m = "rerank" in m

                                ans, docs, t = run_once(
                                    question=q,
                                    vs=vs,
                                    store_dir=store_dir,
                                    k=int(k),
                                    use_mmr=use_mmr_m,
                                    lambda_mult=float(lambda_mult),
                                    use_hybrid=use_hybrid_m,
                                    bm25_k=int(bm25_k),
                                    use_rerank=use_rerank_m
                                )

                                top_src = ""
                                if docs:
                                    d0 = docs[0]
                                    top_src = f"{d0.metadata.get('source','doc')} p.{d0.metadata.get('page','?')}"

                                rows.append({
                                    "question": q,
                                    "mode": m,
                                    "answer": ans,
                                    "top1_source": top_src,
                                    "retrieval_s": round(t["t_retrieval"], 3),
                                    "rerank_s": round(t["t_rerank"], 3),
                                    "llm_s": round(t["t_llm"], 3),
                                    "answer_chars": len(ans or "")
                                })

                                step += 1
                                prog.progress(min(1.0, step / total))

                        import pandas as pd
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results CSV",
                            data=csv_bytes,
                            file_name=f"eval_{index_name}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Evaluation error: {e}")


if __name__ == "__main__":
    main()
