# Personal Wikipedia — RAG Q&A (CLI + Streamlit UI)

A simple Retrieval-Augmented Generation (RAG) app that answers questions strictly from your own document. It supports TXT or PDF and can run as:

- CLI: `main.py`
- Streamlit UI: `app.py` (with a History page under `pages/History.py`)

## Tech stack
- Embeddings: `nomic-embed-text` via Ollama (local)
- Vector store: FAISS (in-memory)
- LLM: Qwen (`qwen/qwen3-32b`) via Groq
- Framework: LangChain + `langchain-ollama` + `langchain-groq`

---

## Project structure
```
personal_wikipedia/
├── app.py                  # Streamlit UI
├── main.py                 # CLI app
├── knowledge_base.txt      # Sample KB (you can upload your own)
├── pages/
│   └── History.py          # Streamlit history page
├── requirements.txt
└── .env                    # Local env vars (not committed)
```

---

## Prerequisites
- Python 3.10+
- Ollama installed and running: https://ollama.ai
- Groq API key: https://console.groq.com/keys

Place your Groq key in a `.env` file at the project root:
```
GROQ_API_KEY="YOUR_API_KEY_HERE"
```

Alternatively, export it in your shell session (see below).

---

## Setup

### Windows (PowerShell)
```powershell
# From the project root (folder that contains requirements.txt)
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare the embedding model (first time only)
ollama pull nomic-embed-text

# Optional: set the key for this session if not using .env
$env:GROQ_API_KEY = "YOUR_REAL_GROQ_API_KEY"
```

### macOS / Linux (bash/zsh)
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare the embedding model (first time only)
ollama pull nomic-embed-text

# Optional: set the key for this session if not using .env
export GROQ_API_KEY="YOUR_REAL_GROQ_API_KEY"
```

---

## Run

### Streamlit UI
```bash
# From the project root (any OS)
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501), upload a `.txt` or `.pdf`, and ask questions.

### CLI app
```bash
# From the project root (any OS)
python main.py
```
Type your question; type `exit` to quit.

---

## How it works
1. Load your file (or upload via UI) and split into chunks (size 1000, overlap 200).
2. Create embeddings for chunks using `nomic-embed-text` via Ollama (runs locally).
3. Build an in-memory FAISS index and use it to retrieve relevant chunks for a question.
4. Feed retrieved context + question to Groq’s Qwen model (`qwen/qwen3-32b`) with a strict prompt that refuses to invent facts.
5. Display only the final answer. The UI hides any hidden reasoning (`<think>`/similar tags) and shows code blocks with copy buttons, while plain text renders normally.

Notes:
- Ollama provides embeddings only; Groq generates the final text answer.
- All indexing is in-memory. Re-upload/re-run to refresh.
- PDF support relies on text-based PDFs (for scanned PDFs, consider OCR).

---

## Troubleshooting
- “ollama not found”: Ensure Ollama is installed and on PATH (or run with an absolute path). Make sure the service is running. You can also set `OLLAMA_HOST=http://127.0.0.1:11434`.
- Embedding errors: Ensure `ollama pull nomic-embed-text` succeeded and Ollama is running.
- Import errors: Install requirements inside the same virtual environment you use to run the app.
- Groq auth errors: Confirm `GROQ_API_KEY` is set (via `.env` or shell env var).
- Streamlit not seeing new packages: Restart Streamlit after installing requirements.

---

## Configuration
- Change embedding model: edit `EMBED_MODEL` in `app.py` (and in `main.py` if you use the CLI).
- Change the LLM: in both `app.py` and `main.py`, update the Groq model (e.g., `qwen/qwen2-7b-instruct`).
- Tuning: adjust `chunk_size`, `chunk_overlap`, or retriever parameters as needed.

---

## Security & privacy
- Embeddings are computed locally by Ollama; your documents aren’t sent to a third-party for retrieval.
- The final answer is generated in the cloud by Groq; only the prompt (question + retrieved snippets) is sent.

---

## License
This project is for educational purposes. Review licenses of dependencies (LangChain, FAISS, Streamlit, Ollama, Groq) before production use.
