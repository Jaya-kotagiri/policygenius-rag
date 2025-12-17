# ingest.py

import os
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

# --- 1. CONFIGURATION & SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

DATA_PATH = "data/"
CHROMA_PATH = "chroma_db/"

# --- 2. DATA LOADING ---
def load_documents():
    logging.info(f"Loading documents from {DATA_PATH}...")

    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    docx_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True
    )

    documents = pdf_loader.load() + docx_loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    logging.info("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks


# --- 3. EMBEDDING & STORAGE ---
def build_vector_database(chunks):
    logging.info("Building vector database with local embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    vectorstore.persist()
    logging.info(f"Vector DB saved at {CHROMA_PATH}")


# --- 4. MAIN ---
def main():
    if not os.path.exists(DATA_PATH):
        logging.error("Data folder not found. Create 'data/' and add documents.")
        return

    documents = load_documents()
    if not documents:
        logging.warning("No documents found. Exiting.")
        return

    chunks = split_documents(documents)
    build_vector_database(chunks)

    logging.info("✅ Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()
