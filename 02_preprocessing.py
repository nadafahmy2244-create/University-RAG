from langchain_community.document_loaders import PyPDFLoader

pdf_path = "documents/msa-university-faculty-of-pharmacy-Student-Handbook.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# إزالة المسافات والأسطر الفارغة
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split())

print(documents[0].page_content[:500])