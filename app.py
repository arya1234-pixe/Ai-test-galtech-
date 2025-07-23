import streamlit as st
import os
from dotenv import load_dotenv
from backend.pdf_processor import process_pdf
from backend.agent_builder import build_agent

load_dotenv()
os.environ["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"] = os.getenv("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")

st.set_page_config(page_title="AI Agent PDF Chat", layout="wide")
st.title("📄 Chat with Your PDF using AI Agent")

# Upload Section
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if not os.path.exists("data/uploaded_pdfs"):
        os.makedirs("data/uploaded_pdfs")
    
    pdf_path = os.path.join("data/uploaded_pdfs", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ PDF uploaded successfully!")

    # Process PDF
    with st.spinner("Processing PDF and building RAG..."):
        process_pdf(pdf_path)
        st.success("✅ PDF processed and vectorized!")

    st.session_state["agent"] = build_agent()

# Chat UI
if "agent" in st.session_state:
    st.subheader("💬 Ask questions about the document")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            response = st.session_state["agent"].run(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Agent", response))

    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
