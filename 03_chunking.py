from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "documents/msa-university-faculty-of-pharmacy-Student-Handbook.pdf"

loader = PyPDFLoader(pdf_path)
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

print(f"Number of chunks: {len(chunks)}")