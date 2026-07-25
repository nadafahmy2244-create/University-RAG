import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(dotenv_path=".env")
print(os.getenv("OPENROUTER_API_KEY"))
from dotenv import dotenv_values

print(dotenv_values(".env"))
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

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

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

query = "What is the attendance policy?"

docs = retriever.invoke(query)

context = docs[0].page_content

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model=os.getenv("OPENROUTER_MODEL"),
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)