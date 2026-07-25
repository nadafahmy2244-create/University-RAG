import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🧠",
    layout="wide"
)


# =========================
# Custom Styling
# =========================

st.markdown(
    """
    <style>

    .stApp {
        background: radial-gradient(
            circle at top right,
            #1e1b4b,
            #020617 60%
        );
    }


    .main-title {
        font-size: 55px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #c084fc,
            #22d3ee
        );
        -webkit-background-clip: text;
        color: transparent;
    }


    .subtitle {
        color:#94a3b8;
        font-size:20px;
    }


    .feature-card {

        background:
        linear-gradient(
            145deg,
            rgba(139,92,246,.25),
            rgba(6,182,212,.15)
        );

        border:1px solid rgba(139,92,246,.5);

        padding:20px;

        border-radius:20px;

        margin:20px 0;

        box-shadow:
        0 0 25px rgba(139,92,246,.25);

    }


    div[data-testid="stChatMessage"] {

        background:
        linear-gradient(
            145deg,
            rgba(30,41,59,.8),
            rgba(15,23,42,.9)
        );

        border-radius:20px;

        border:1px solid rgba(99,102,241,.4);

        padding:15px;

    }


    .source-box {

        border:1px solid #06b6d4;

        padding:15px;

        border-radius:15px;

        background:#020617;

        box-shadow:
        0 0 15px rgba(6,182,212,.4);

    }


    </style>

    """,
    unsafe_allow_html=True
)

# =========================
# Load API
# =========================

load_dotenv()


api_key = os.getenv("OPENROUTER_API_KEY")


if not api_key:

    api_key = st.secrets["OPENROUTER_API_KEY"]


model = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)



# =========================
# Load Vector Database
# =========================

@st.cache_resource
def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )


    return vectorstore.as_retriever(
        search_kwargs={
            "k":3
        }
    )


retriever = load_retriever()



# =========================
# Sidebar
# =========================


with st.sidebar:

    st.markdown(
        """
        # 🧠 RAG AI Assistant

        Intelligent document
        question answering system.
        """
    )


    st.divider()


    st.markdown(
        """
        ### ⚙️ Technology Stack

        🦜 LangChain  
        🗄 ChromaDB  
        🤗 HuggingFace Embeddings  
        ⚡ GPT-4o-mini  
        🚀 Streamlit Cloud
        """
    )


    st.divider()


    language = st.selectbox(
        "🌍 Answer Language",
        [
            "English",
            "Arabic"
        ]
    )


    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()



# =========================
# Header
# =========================


st.markdown(
"""
<div class="main-title">
🧠 University AI Assistant
</div>

<div class="subtitle">
Powered by Retrieval Augmented Generation ✨
</div>
""",
unsafe_allow_html=True
)



st.markdown(
    """
    <div class="card">

    🚀 This assistant uses:

    • Document Retrieval  
    • Semantic Search  
    • Vector Database  
    • Large Language Models  

    </div>
    """,
    unsafe_allow_html=True
)



# =========================
# Chat Memory
# =========================


if "messages" not in st.session_state:

    st.session_state.messages = []



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])



# =========================
# User Question
# =========================


question = st.chat_input(
    "Ask your question..."
)



if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.write(question)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching knowledge base..."
        ):


            docs = retriever.invoke(
                question
            )


            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )



            if language == "Arabic":

                lang_instruction = """
                Answer in Arabic.
                """

            else:

                lang_instruction = """
                Answer in English.
                """



            prompt = f"""

            {lang_instruction}

            Answer only using the context below.

            Context:

            {context}


            Question:

            {question}

            """



            response = client.chat.completions.create(

                model=model,

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]

            )



            answer = (
                response
                .choices[0]
                .message
                .content
            )



            st.write(answer)



            st.markdown(
                """
                <div class="source">

                📚 Retrieved Source

                </div>
                """,
                unsafe_allow_html=True
            )


            for doc in docs:

                st.caption(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                )



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )