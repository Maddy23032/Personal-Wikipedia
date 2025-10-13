"""
Personal Wikipedia - RAG Q&A bot using LangChain, FAISS, Ollama embeddings, and Groq LLM.

Follow the numbered comments to see how each step in the instructions is implemented.
"""

# Step 1: Imports and Setup
from __future__ import annotations
import os
from dotenv import load_dotenv

# LangChain core & community
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Groq LLM
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()


def require_env_var(name: str) -> str:
    """Utility to fetch an environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value or value == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            f"Required environment variable '{name}' is missing or placeholder. "
            "Please set it in .env before running."
        )
    return value


# Step 3: Define the RAG Logic in a main() function
def main() -> None:
    # Step 2: Load API Key and validate
    groq_api_key = require_env_var("GROQ_API_KEY")

    # Step 4: Load the Document
    # Use TextLoader to load the content from knowledge_base.txt
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"Knowledge base file not found at {kb_path}. Ensure it exists."
        )
    loader = TextLoader(kb_path, encoding="utf-8")
    docs = loader.load()

    # Step 5: Split the Document into Chunks
    # Using chunk_size=1000 and chunk_overlap=200
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Step 6: Create Embeddings and the Vector Store
    # Uses HuggingFace embeddings (free, no external service needed)
    # First run will download the model (~80MB) from HuggingFace
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize HuggingFaceEmbeddings. Ensure sentence-transformers is installed: pip install sentence-transformers"
        ) from e

    # Build an in-memory FAISS vector store from the chunks using the embeddings
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # Step 7: Create the Retriever
    retriever = vector_store.as_retriever()

    # Step 8: Define the Prompt Template
    template = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer. \n\n"
        "Context: {context} \n\n"
        "Question: {input} \n\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(template)

    # Helper to format retrieved docs into a single context string
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Step 9: Initialize the LLM (Groq)
    # ChatGroq reads GROQ_API_KEY from env; pass the model via `model`
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.3,
    )

    # Step 10: Create the RAG Chain using LCEL
    # Chain: {"context": retriever | format_docs, "input": passthrough} | prompt | llm
    rag_chain = (
        {"context": itemgetter("input") | retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Step 11: Implement the User Interaction Loop
    print("Personal Wikipedia (RAG) — ask about 'Peravallur'. Type 'exit' to quit.\n")
    while True:
        try:
            user_q = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # Invoke the chain
        try:
            response = rag_chain.invoke({"input": user_q})
            # ChatGroq returns a ChatMessage-like object; get the content
            answer = getattr(response, "content", str(response))
        except Exception as e:
            answer = f"Error while generating answer: {e}"

        print(f"\nAnswer: {answer}\n")


# Step 12: Entry Point
if __name__ == "__main__":
    main()
