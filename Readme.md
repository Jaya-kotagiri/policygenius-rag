# PolicyGenius RAG Bot ⚖️

A **production-style Retrieval-Augmented Generation (RAG)** system for answering HR policy questions **strictly from company documents**, with observability via **LangSmith** and guardrails to prevent hallucinations.

This project is designed to demonstrate **real-world RAG engineering skills** suitable for mid-to-senior ML / AI roles.

---

## 🚀 What This Bot Does

- Answers **only HR policy questions** using provided documents
- Refuses to answer personal, conversational, or out-of-scope queries
- Avoids hallucination by:
  - Strict prompting
  - Retrieval-first architecture
  - Zero-temperature LLM
- Provides **auditable traces** using LangSmith

---

## 🧠 Architecture Overview

```
User Question
     ↓
Retriever (Chroma Vector DB)
     ↓
Relevant Policy Chunks
     ↓
Strict Prompt + LLM (Groq LLaMA 3.3 70B)
     ↓
Final Answer (with citation)
```

---

## 📁 Project Structure

> **Data governance note:** Raw HR policy documents inside the `data/` directory are intentionally excluded from version control to prevent accidental sharing of proprietary or confidential company information.



```
policygenius-rag/
│
├── app.py                 # Streamlit app (chat UI + RAG chain)
├── ingest.py              # Document ingestion & vector DB builder
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore
│
├── data/                  # HR policy PDFs / DOCX (ignored by git)
├── chroma_db/             # Vector database (ignored by git)
└── .env                   # API keys (ignored by git)
```

---

## 🔐 Environment Variables (`.env`)

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=policygenius-rag
```

---

## 📦 Dependencies

See `requirements.txt`. Core libraries:

- `streamlit` – UI
- `langchain` ecosystem – RAG framework
- `chromadb` – vector storage
- `sentence-transformers` – embeddings
- `langchain-groq` – LLM inference
- `python-dotenv` – environment management

---

## 🧾 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add policy documents

Place PDF / DOCX files inside:

```text
data/
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

### 4️⃣ Index documents

Use the **Admin Sidebar → Re-index Documents** button.

---

## 🔍 Observability (LangSmith)

All RAG steps are traced:

- Retriever calls
- Prompt construction
- LLM response

You can inspect runs at:

👉 https://smith.langchain.com

This allows you to **prove**:
- Which documents were retrieved
- Why a specific answer was generated
- That the model did not hallucinate

---

## 🧪 Design Decisions (Interview-Ready)

- **Temperature = 0** → deterministic answers
- **Strict prompt rules** → no guessing, no role-play
- **Header-aware chunking** → accurate section citations
- **Vector DB reset on re-ingestion** → avoids stale data

---

## 🛑 Non-Goals

- Not a general chatbot
- Not trained on internal data
- No fine-tuning (pure RAG)

---

## 📌 Future Improvements

- Hybrid Search (BM25 + Vector)
- Query classification layer
- Role-based access control
- Evaluation dataset + automated testing

---

## 👩‍💻 Author

Built as a **portfolio-grade RAG system** to demonstrate applied LLM engineering, not a toy demo.

