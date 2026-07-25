from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("documents/msa-university-faculty-of-pharmacy-Student-Handbook.pdf")
documents = loader.load()

# Preprocessing
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split())

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")