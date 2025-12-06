import os
import time
import streamlit as st
from dotenv import load_dotenv

# Google GenAI system config
from google.generativeai import configure

# LangChain Core
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader

# LangChain Google GenAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# Vector Store
from langchain.vectorstores import FAISS


# ==========================================================
#   ENV + GOOGLE API CONFIG
# ==========================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing! Add it to your .env file first.")
    st.stop()

configure(api_key=API_KEY)  # Forces API key auth (avoids ADC errors)


# ==========================================================
#   LLM INITIALIZER — Gemini 2.5 Flash
# ==========================================================
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        convert_system_message_to_human=True,
    )


# ==========================================================
#   DOCUMENT COLLECTION
# ==========================================================
def load_web_documents(urls: list):
    loader = WebBaseLoader(urls)
    return loader.load()


# ==========================================================
#   SPLIT DOCUMENTS INTO CHUNKS
# ==========================================================
def split_documents(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", ","],
    )
    return splitter.split_documents(data)


# ==========================================================
#   FAISS VECTORSTORE BUILDER (NO PICKLE)
# ==========================================================
def build_vectorstore(docs, folder_path="faiss_store"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(folder_path)
    return vectorstore


# ==========================================================
#  LOAD VECTORSTORE (REQUIRES EMBEDDINGS)
# ==========================================================
def load_vectorstore(folder="faiss_store"):
    if not os.path.exists(folder):
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return FAISS.load_local(
        folder,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

# ==========================================================
#  QA CHAIN
# ==========================================================
def build_qa_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        verbose=False
    )


# ==========================================================
#  RUN QA — `.invoke()` (2025 standard)
# ==========================================================
def get_answer(qa_chain, query: str):
    return qa_chain.invoke({"query": query})


# ==========================================================
#  STREAMLIT UI
# ==========================================================
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

llm = initialize_llm()
vectorstore_folder = "faiss_store"

# Initialize session state for dynamic URL inputs
if "url_count" not in st.session_state:
    st.session_state.url_count = 1

# Function to add more URL inputs
def add_url():
    st.session_state.url_count += 1

# URL inputs - dynamic
urls = []
for i in range(st.session_state.url_count):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

# Plus button to add more URLs
st.sidebar.button("➕ Add another URL", on_click=add_url)

process_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()


# ==========================================================
#   PROCESS URLS
# ==========================================================
if process_clicked:
    urls = [u.strip() for u in urls if u.strip()]

    if not urls:
        st.error("⚠️ Please enter at least one valid URL")
    else:
        try:
            main_placeholder.text("🔍 Fetching articles...")
            docs_raw = load_web_documents(urls)

            main_placeholder.text("✂️ Splitting content...")
            docs = split_documents(docs_raw)

            main_placeholder.text("🔢 Generating embeddings & building FAISS index...")
            build_vectorstore(docs, vectorstore_folder)

            time.sleep(1)
            main_placeholder.text("🚀 Done! Your research index is ready. Ask below.")

        except Exception as e:
            st.error(f"❌ Error while processing: {str(e)}")


# ==========================================================
#   ANSWER DISPLAY AREA
# ==========================================================
answer_placeholder = st.container()

# ==========================================================
#  FOOTER WITH INPUT - FIXED AT BOTTOM
# ==========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-bottom: 150px;
        }
        .fixed-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #0e1117;
            padding: 15px 50px;
            z-index: 999;
        }
        .fixed-footer label {
            color: #fafafa;
            font-size: 14px;
            margin-bottom: 8px;
            display: block;
        }
        .fixed-footer input {
            width: 100%;
            padding: 12px;
            background-color: #262730;
            border: 1px solid #444;
            border-radius: 5px;
            color: white;
            font-size: 14px;
        }
        .fixed-footer hr {
            margin-top: 15px;
            border: none;
            border-top: 1px solid #444;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Footer input
query = st.text_input("Ask anything about the collected news:", key="query_input")

# Bottom line
st.markdown("<hr style='margin-top: 10px; border: none; border-top: 1px solid #444;'>", unsafe_allow_html=True)

if query:
    vectorstore = load_vectorstore(vectorstore_folder)

    if not vectorstore:
        st.warning("⚠️ Process URLs first to build search index!")
    else:
        try:
            qa_chain = build_qa_chain(llm, vectorstore)

            with st.spinner("🔎 Searching & analyzing..."):
                answer = get_answer(qa_chain, query)

            with answer_placeholder:
                st.header("📌 Answer")
                st.write(answer["result"])

        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
