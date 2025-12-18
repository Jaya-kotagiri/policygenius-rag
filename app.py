# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Professional check for API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

CHROMA_PATH = "chroma_db/"

@st.cache_resource
def load_retriever():
    # Note: For 15 LPA, mention you considered 'bge-large-en' for better accuracy
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def load_llm():
    # UPGRADED: Moving to the 70B model for superior reasoning
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        section = doc.metadata.get("section", "N/A")
        heading = doc.metadata.get("heading", "N/A")
        formatted.append(f"[Source: {source} | Sec {section} - {heading}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(retriever, llm):
    # HARDENED PROMPT: Prevents logic errors like "Paid Loss of Pay"
    prompt = ChatPromptTemplate.from_template("""
    SYSTEM ROLE:
    You are a professional HR Policy Audit Assistant.

    STRICT OPERATING RULES:

    1. SCOPE CONTROL
    - Answer ONLY questions related to HR policies using the provided context.
    - If the question is personal, conversational, or outside HR policy scope, respond EXACTLY with:
    "I could not find this information in the policy documents."
    - Do NOT include sources when responding with the above message.

    2. EVIDENCE CONTROL
    - Use ONLY information explicitly stated in the provided documents.
    - Do NOT infer, assume, speculate, or fill gaps using common sense.
    - If the requested information, category, or entitlement is not explicitly defined in the policy, clearly state that it is not defined.

    3. ANSWER QUALITY
    - Provide one clear, direct answer.
    - Combine related information into a single response when applicable.
    - Avoid repetition, commentary, or defensive explanations.

    4. STRUCTURE & CLARITY
    - If the user asks to list or enumerate items, but the policy does not provide explicit items, state that the policy specifies a count or rule but does not list individual items.
    - Do NOT invent names, examples, or classifications.
    - Keep the answer format neat and clear. no stuffed, messy paragraphs.

    5. LOGICAL CONSISTENCY
    - Do not contradict the policy wording.
    - Do not reframe restrictions or exclusions as benefits.

    6. CITATION
    - Cite the document name and section number ONCE at the end of the answer.
    - Do not cite sources if the answer is not found.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    st.set_page_config(page_title="PolicyGenius", page_icon="⚖️")
    st.title("🤖 PolicyGenius HR Chatbot")

    # -------------------------------
    # 🔁 RESTORED: ADMIN SIDEBAR
    # -------------------------------
    with st.sidebar:
        st.header("⚙️ Admin Controls")
        st.info("Place your PDF/DOCX files in the 'data/' folder.")
        if st.button("🔄 Re-index Documents"):
            from ingest import ingest_documents # Dynamic import to avoid circular issues
            with st.spinner("Processing documents..."):
                ingest_documents()
            st.cache_resource.clear()
            st.success("Vector Database Updated!")

    # Load Components
    try:
        retriever = load_retriever()
        llm = load_llm()
        rag_chain = build_rag_chain(retriever, llm)
    except Exception as e:
        st.warning("Vector DB not found. Please click 'Re-index' in the sidebar.")
        return

    # Chat Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about company policy..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()