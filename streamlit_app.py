import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

st.set_page_config(page_title="University RAG", page_icon="🎓")

st.title("🎓 University RAG Assistant")

question = st.text_input("Ask a question about the student handbook")

if question:

    docs = retriever.invoke(question)

    context = docs[0].page_content

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL"),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    st.subheader("Answer")
    st.write(response.choices[0].message.content)

    st.subheader("Source")
    st.write(docs[0].metadata["source"])