"""
Streamlit UI for Personal Wikipedia RAG App
- Upload TXT or PDF
- Builds FAISS index with Ollama embeddings (nomic-embed-text)
- Answers via Groq (Qwen qwen/qwen3-32b)
- Shows response with copy & download buttons
- History page via Streamlit multipage (pages/History.py)
"""
from __future__ import annotations
import os
import io
import json
import re
from pathlib import Path
from typing import List

import streamlit as st
from streamlit.components.v1 import html
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from operator import itemgetter

# Load env
load_dotenv()

APP_TITLE = "Personal Wikipedia (Streamlit)"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3-32b"


def require_env_var(name: str) -> str:
    # Try environment variable first
    val = os.getenv(name)
    
    # If not found, try Streamlit secrets (for cloud deployment)
    if not val or val == "YOUR_API_KEY_HERE":
        try:
            val = st.secrets.get(name)
        except (AttributeError, FileNotFoundError, KeyError):
            pass
    
    if not val or val == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            f"Missing {name}. Add it to Streamlit Cloud Secrets (Settings → Secrets) or set it in your .env file."
        )
    return val


def _load_docs_from_uploaded(file):
    # Persist uploaded file to a temp path for loaders
    suffix = Path(file.name).suffix.lower()
    tmp_path = Path(st.session_state.get("upload_dir", ".")) / f"_kb{suffix}"
    tmp_path.write_bytes(file.getvalue())

    if suffix == ".pdf":
        loader = PyPDFLoader(str(tmp_path))
        return loader.load()
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(tmp_path))
        return loader.load()
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(tmp_path), encoding="utf-8")
        return loader.load()
    else:
        raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx")


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def _build_retriever(docs):
    # Uses HuggingFace embeddings (free, runs locally via sentence-transformers)
    # Set model_kwargs to ensure proper device handling for deployment
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vs = FAISS.from_documents(docs, embeddings)
    return vs.as_retriever()


def _make_chain(retriever):
    prompt_tmpl = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer. \n\n"
        "Context: {context} \n\n"
        "Question: {input} \n\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(prompt_tmpl)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # ChatGroq reads GROQ_API_KEY from environment; pass the model via `model`
    require_env_var("GROQ_API_KEY")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
    )

    # Ensure the retriever gets the string question, not the entire input dict
    chain = ({"context": itemgetter("input") | retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm)
    return chain


def _init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {question, answer}
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = str(Path.cwd() / ".streamlit_uploads")
        Path(st.session_state.upload_dir).mkdir(parents=True, exist_ok=True)


def _download_bytes(filename: str, content: str) -> bytes:
    buf = io.BytesIO()
    buf.write(content.encode("utf-8"))
    buf.seek(0)
    return buf.getvalue()


# Removed explicit copy button; we'll use st.code which includes a copy button.


def _strip_think_blocks(text: str) -> str:
    """Remove hidden reasoning blocks like <think>...</think> (or similar) from text before display.
    Handles case-insensitive tags and trims extra whitespace after removal.
    """
    if not text:
        return text
    patterns = [
        r"(?is)<think>.*?</think>",
        r"(?is)<reasoning>.*?</reasoning>",
        r"(?is)<chain[-_ ]?of[-_ ]?thought>.*?</chain[-_ ]?of[-_ ]?thought>",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned)
    # Remove any stray opening/closing tags if present
    cleaned = re.sub(r"(?is)</?think>|</?reasoning>|</?chain[-_ ]?of[-_ ]?thought>", "", cleaned)
    return cleaned.strip()


def _split_answer_blocks(text: str):
    """Split an answer into a sequence of blocks: [{'type':'text','content':...}, {'type':'code','content':...,'lang':...}].
    Recognizes fenced code blocks delimited by triple backticks, with optional language tag.
    """
    if not text:
        return []
    pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*\n(.*?)```", re.DOTALL)
    pos = 0
    blocks = []
    for m in pattern.finditer(text):
        start, end = m.span()
        # preceding text
        if start > pos:
            pre = text[pos:start]
            if pre.strip():
                blocks.append({"type": "text", "content": pre.strip()})
        lang = m.group(1).strip() or None
        code = m.group(2)
        blocks.append({"type": "code", "content": code.rstrip(), "lang": lang})
        pos = end
    # trailing text
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            blocks.append({"type": "text", "content": tail.strip()})
    return blocks


def _render_answer(answer: str):
    blocks = _split_answer_blocks(answer)
    if not blocks:
        st.write("")
        return
    for b in blocks:
        if b["type"] == "code":
            st.code(b["content"], language=b.get("lang") or None)
        else:
            st.write(b["content"])  # render normal text without code block


# Streamlit app
_defrag_css = """
<style>
    .small-note { color: #666; font-size: 0.9em; }
</style>
"""


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="centered")
    _init_session()
    st.markdown(_defrag_css, unsafe_allow_html=True)

    st.title(APP_TITLE)
    st.write("Upload a .txt or .pdf knowledge base, ask questions, and get answers only from your file.")

    with st.sidebar:
        st.header("Settings")
        st.write("Embeddings: all-MiniLM-L6-v2 (via HuggingFace, free)")
        st.write("LLM: Qwen (qwen/qwen3-32b via Groq)")
        st.markdown("<div class='small-note'>First run downloads the embedding model (~80MB).</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload knowledge base (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=False)

    retriever = None
    if uploaded is not None:
        try:
            docs = _load_docs_from_uploaded(uploaded)
            chunks = _split_docs(docs)
            retriever = _build_retriever(chunks)
            st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

    # Question input and response area
    user_q = st.text_input("Your question")
    ask_btn = st.button("Ask")

    if ask_btn:
        if not retriever:
            st.warning("Please upload a knowledge base first.")
        elif not user_q.strip():
            st.warning("Please enter a question.")
        else:
            try:
                chain = _make_chain(retriever)
                resp = chain.invoke({"input": user_q})
                answer_raw = getattr(resp, "content", str(resp))
                answer = _strip_think_blocks(answer_raw)
                st.session_state.history.append({"question": user_q, "answer": answer})

                st.subheader("Answer")
                _render_answer(answer)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Answer (.txt)",
                        data=_download_bytes("answer.txt", answer),
                        file_name="answer.txt",
                        mime="text/plain",
                    )
            except Exception as e:
                st.error(f"Error while generating answer: {e}")

    st.divider()
    st.write("Go to the History page (left sidebar: Pages) to see past Q&A.")


if __name__ == "__main__":
    main()
