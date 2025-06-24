import os
import tempfile
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "llm_translator"

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
dimension = len(embeddings.embed_query("test"))

st.set_page_config(page_title="📄 LLM Q&A from PDF/Web", page_icon="🤖")
st.title("📄 Ask Questions from PDF or Web Content")

uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])
web_url = st.text_input("🌐 Or enter a web URL")
query = st.text_input("🔎 Ask a question about the content")

def load_docs(pdf_file=None, url=None):
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            loader = PyPDFLoader(tmp.name)
    elif url:
        loader = WebBaseLoader(url)
    else:
        return []
    return loader.load()

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    import faiss
    index = faiss.IndexFlatL2(dimension)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    ids = [str(uuid4()) for _ in range(len(chunks))]
    store.add_documents(documents=chunks, ids=ids)
    return store

if "vectorstore" not in st.session_state:
    if uploaded_file or web_url:
        with st.spinner("📄 Processing document..."):
            docs = load_docs(uploaded_file, web_url)
            chunks = chunk_docs(docs)
            st.session_state.vectorstore = build_vectorstore(chunks)
            st.success("✅ Document processed and index built.")


if query:
    if "vectorstore" not in st.session_state:
        st.warning("Please upload a file or enter a URL first.")
    else:
        with st.spinner(" Thinking..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever()
            )
            try:
                answer = qa_chain.run(query)
                st.success(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
