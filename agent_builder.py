from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent
from backend.tools import get_tools
import os

def load_vectorstore(index_path='vectordb/faiss_index'):
    return FAISS.load_local(index_path, embeddings=ChatOpenAI().embedding)

def build_agent():
    llm = ChatOpenAI(temperature=0)

    tools = get_tools()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    retriever = load_vectorstore().as_retriever(search_type="similarity", k=3)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )

    return qa_chain
