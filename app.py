import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os

# ---- Set Up ----
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot")

# ---- Upload PDF ----
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    # Save temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    # Load and split text
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Question Input
    st.success("PDF Processed! Now ask a question 👇")
    query = st.text_input("Ask something about the PDF:")

    if query:
        # QA Chain
       from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 256},
    task="text2text-generation",  # ✅ This is the required fix
    huggingfacehub_api_token="your_token_here"  # optional if set via env variable
)
