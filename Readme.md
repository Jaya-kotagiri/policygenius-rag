
PolicyGenius: HR Policy RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) application built to provide precise, cited answers from company HR documents. This project addresses the metadata dilution problem commonly found in naive RAG systems by using header-aware chunking to preserve semantic and structural context.

------------------------------------------------------------

Key Features

Deterministic Reasoning
Powered by Llama-3.3-70B-Versatile (via Groq) with temperature set to zero for audit-safe, deterministic responses.

Smart Chunking
Custom ingestion pipeline using regex-based header detection (for example, 7.2.1 Leave Policy) to ensure each response cites the exact governing clause.

Production Observability
Integrated with LangSmith to trace retrieval accuracy, latency, and end-to-end reasoning behavior.

Dynamic Re-indexing
Streamlit-based administrative control to refresh the vector database without restarting the application.

------------------------------------------------------------

System Architecture

1. Document Ingestion
PDF and DOCX files are parsed from the data directory.

2. Metadata Enrichment
Each chunk is enriched with source file name, page number, and policy section header.

3. Vector Storage
ChromaDB is used with all-MiniLM-L6-v2 embeddings for efficient semantic retrieval.

4. Grounded Generation
A system-level auditor prompt constrains the LLM to answer strictly from retrieved context and refuse unsupported queries.

------------------------------------------------------------

Tech Stack

Framework: LangChain 0.2  
LLM: Meta Llama 3.3 70B  
Vector Database: ChromaDB  
User Interface: Streamlit  
Observability: LangSmith  
Embeddings: HuggingFace Sentence Transformers  

------------------------------------------------------------

Installation and Setup

1. Clone the repository

git clone https://github.com/your-username/policygenius-rag.git  
cd policygenius-rag  

2. Configure environment variables

Create a .env file in the project root with the following entries:

GROQ_API_KEY=your_groq_api_key  
LANGCHAIN_API_KEY=your_langchain_api_key  
LANGCHAIN_TRACING_V2=true  
LANGCHAIN_PROJECT=policygenius-rag  

3. Install dependencies

pip install -r requirements.txt  

------------------------------------------------------------

Usage

Add Documents  
Create a data directory and place HR policy PDF or DOCX files inside it.

Run the Application  
streamlit run app.py  

Initialize Index  
Use the Streamlit sidebar option to re-index documents and build the local ChromaDB.

Query the System  
Example questions:
- What is the policy for relocation grant?
- How many privilege leaves can be carried forward?

------------------------------------------------------------

Technical Decisions and Trade-offs

Model Selection  
Llama-3.3-70B was chosen to handle complex logical dependencies in legal and HR text that smaller models frequently hallucinate.

Header-Aware Splitting  
Ensures that section titles remain tightly coupled with their content, enabling precise citation and compliance-grade answers.

Zero Temperature  
For HR and policy systems, creativity introduces risk. Deterministic decoding improves groundedness and auditability.

------------------------------------------------------------

Roadmap

Hybrid search using BM25 for keyword-heavy queries  
Cross-encoder re-ranking to improve top-1 retrieval accuracy  
Local LLM support via Ollama for full data privacy  

------------------------------------------------------------

Author

Jayalaxmi Kotagiri  
LinkedIn: https://www.linkedin.com/in/jaya-kotagiri/
