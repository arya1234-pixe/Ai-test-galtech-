from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load your document (replace with your actual file)
loader = TextLoader("data/sample.txt")  # path to your file
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings
embedding_model = OpenAIEmbeddings()

# Create FAISS index
db = FAISS.from_documents(docs, embedding_model)

# Save to disk
db.save_local("vectordb/faiss_index")
print("✅ FAISS index saved.")
