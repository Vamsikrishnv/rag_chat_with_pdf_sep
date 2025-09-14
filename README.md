RAG Chat with PDFs

An interactive Retrieval-Augmented Generation (RAG) app built with LangChain, FAISS, and Streamlit.
Upload your PDFs and ask questions — get precise answers with citations.

🚀 Features

📂 PDF Upload & Indexing – add personal or work PDFs easily

🧩 Chunking Strategies – recursive, fixed, or token-based splits

🔍 Retrieval Modes – Similarity, MMR (diverse), Hybrid (BM25 + embeddings)

🎯 Cross-Encoder Rerank – improves answer relevance

📊 Evaluation Tab – compare retrieval modes across questions

💾 Persistent FAISS Storage – incremental updates or full reindex

💬 Chat UI – answers with citations, sources table, CSV export

🏗 Architecture

Streamlit – UI for chat & controls

LangChain – document loading, splitting, retrievers

FAISS – vector database for embeddings

BM25 Retriever – sparse keyword search

Hybrid Fusion (RRF) – merges dense + sparse results

OpenAI / HuggingFace Embeddings – flexible embeddings provider

⚙️ Setup
1. Clone the repo
git clone https://github.com/Vamsikrishnv/rag_chat_with_pdf_sep.git
cd rag_chat_with_pdf_sep

2. Create a virtual environment
python -m venv .venv
# Activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Configure API Keys

Create a .env file (already .gitignored):

OPENAI_API_KEY=your-key-here
HF_MODEL=sentence-transformers/all-MiniLM-L6-v2

5. Run Streamlit app
streamlit run streamlit_app.py


👉 Open http://localhost:8501
 in your browser.

🖼️ Demo


🧪 Example Usage

Upload your resume PDF

Ask: “What projects demonstrate RAG expertise?”

Get an answer with citations pointing to exact pages

🔮 Next Up

 Multi-file cross-PDF search

 Auth & user sessions

 Cloud deployment (Streamlit Cloud / Docker)

👨‍💻 Author

Vamshi Krishna
Full-Stack & GenAI Engineer |
