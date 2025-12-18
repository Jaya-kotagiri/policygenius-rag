
import os
import shutil
import logging
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

DATA_PATH = "data/"
CHROMA_PATH = "chroma_db/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_documents():
    logging.info(f"📄 Loading documents from {DATA_PATH}...")
    documents = []
    
    # PDF Loader
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    for doc in pdf_loader.load():
        doc.metadata = {"source": os.path.basename(doc.metadata.get("source", "Unknown")), "page": doc.metadata.get("page", 0)}
        documents.append(doc)

    # DOCX Loader
    docx_loader = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    for doc in docx_loader.load():
        doc.metadata = {"source": os.path.basename(doc.metadata.get("source", "Unknown")), "page": "N/A"}
        documents.append(doc)

    return documents

def split_documents(documents):
    logging.info("Splitting documents with Header-Aware Metadata...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_chunks = []

    for doc in documents:
        # Split text into sections based on numbering like "15.2 Casual Leave"
        # This regex handles "1.0", "15.2.1", etc.
        sections = re.split(r"(\n\d+\.\d+(?:\.\d+)?\s+[A-Z].*)", doc.page_content)
        
        current_section = "General/Intro"
        current_heading = "N/A"

        for i in range(len(sections)):
            part = sections[i].strip()
            if not part: continue

            # If this part is a header, update the metadata for the NEXT part
            header_match = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.*)", part)
            if header_match:
                current_section = header_match.group(1)
                current_heading = header_match.group(2)
                continue # The text following this header is in the next iteration

            # Create chunks for this specific section content
            chunks = splitter.create_documents(
                [part], 
                metadatas=[{**doc.metadata, "section": current_section, "heading": current_heading}]
            )
            final_chunks.extend(chunks)

    logging.info(f"Created {len(final_chunks)} chunks with unique section mapping.")
    return final_chunks

def build_vector_database(chunks):
    logging.info("🧠 Building vector database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    logging.info(f"✅ Vector DB saved at {CHROMA_PATH}")

def ingest_documents():
    if not os.path.exists(DATA_PATH): return
    docs = load_documents()
    if not docs: return
    chunks = split_documents(docs)
    build_vector_database(chunks)

if __name__ == "__main__":
    ingest_documents()