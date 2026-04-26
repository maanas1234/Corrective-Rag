# Second Brain CRAG (Corrective RAG)

A personal “second brain” that lets you query your Markdown notes with a **Corrective RAG (CRAG)** pipeline: multi-query retrieval, relevance grading, optional web search, cross-encoder reranking, and a grounded final answer with sources.

---

## 🚀 What this does

* **Ingest notes once** into a local vector index (FAISS)
* **Ask questions** via a LangGraph pipeline:

  1. Multi-query retrieval (LangChain `MultiQueryRetriever`)
  2. Grade/filter irrelevant chunks
  3. If nothing relevant → web search + scrape and append
  4. Cross-encoder rerank → keep top-k
  5. LLM generates answer **grounded in retrieved context**, with sources
* **Frontend** with Streamlit

---

## 📁 Project Structure

```
second-brain/
│
├── app.py              # Streamlit UI
├── rag_graph.py        # LangGraph CRAG pipeline
├── retriever.py        # FAISS loader, retriever, reranker, generator
├── grader.py           # LLM relevance grader (yes/no)
├── ingest.py           # Build FAISS index from Markdown notes
├── web_search.py       # Web search + scraping helpers
│
├── data/
│   └── notes/          # Your Markdown notes
│
└── vectorstore/        # FAISS index (local only)
```

---

## ⚙️ Setup

### 1) Create & activate virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -U pip
pip install streamlit langgraph langchain langchain-community langchain-openai langchain-huggingface sentence-transformers faiss-cpu python-dotenv
```

### 3) Add environment variables

Create a `.env` file in the root:

```env
OPENROUTER_API_KEY=your_key_here
```

---

## 📥 Ingest your notes

Place your Markdown notes inside:

```
second-brain/data/notes/
```

Then run:

```powershell
python ingest.py
```

This builds/updates the FAISS index in:

```
second-brain/vectorstore/
```

---

## 🧠 Run the CRAG pipeline (CLI)

```powershell
python rag_graph.py
```

---

## 🌐 Run the Streamlit UI

```powershell
streamlit run app.py
```

If you encounter:

```
ModuleNotFoundError: No module named 'torchvision'
```

Run Streamlit with file-watcher disabled:

```powershell
streamlit run app.py --server.fileWatcherType none
```

---

## 🔒 Notes on Safety

* `vectorstore/index.pkl` uses pickle
* Loading FAISS with `allow_dangerous_deserialization=True` can be unsafe
* Treat `vectorstore/` as **local-only**
* Do NOT commit model/index artifacts

---

## 🚫 .gitignore (Recommended)

```
venv/
__pycache__/
*.pyc
second-brain/vectorstore/
```

(Add any private note folders if needed)

---

## 📄 License

Add a license of your choice (MIT, Apache-2.0, etc.)

---

## 📝 One-liner Description

> A personal Second Brain built with a Corrective RAG (CRAG) pipeline: multi-query retrieval, grading, optional web search, cross-encoder reranking, and a Streamlit UI for querying Markdown notes.
