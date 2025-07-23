import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


# Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key="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")


# Title
st.title("📄 PDF Chatbot")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Session state to store chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Step 2: Load and split
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Step 3: Create vectorstore
    vectordb = FAISS.from_documents(docs, embedding_model)

    # Step 4: Setup retrieval chain
    llm = ChatOpenAI(openai_api_key="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")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever(), memory=memory)
    st.session_state.qa_chain = qa_chain

    st.success("✅ PDF processed. You can now ask questions below.")

# Step 5: Input box for questions
if st.session_state.qa_chain:
    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        result = st.session_state.qa_chain.run(user_question)
        st.markdown(f"**Answer:** {result}")
