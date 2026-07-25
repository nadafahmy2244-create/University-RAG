from langchain_community.document_loaders import PyPDFLoader

pdf_path = "documents/msa-university-faculty-of-pharmacy-Student-Handbook.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"Number of pages: {len(documents)}")
