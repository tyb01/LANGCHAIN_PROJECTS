import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec, Vector
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import uuid

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["HUGGING_FACE_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

INDEX_NAME = "hybrid-search-legal-app"
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

model_name = "all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)
bm25_encoder = BM25Encoder().default()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
prompt = ChatPromptTemplate.from_template(
                """
                You are a legal assistant. Answer the question based only on the document context below.
                Be clear and concise, simplifying legal terms where necessary.

                Context:
                {context}

                Question: {input}
                """
            )
st.title("Legal Document RAG App with Hybrid Search + LLM")
st.markdown("Upload a legal or policy PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

def vector_embedding():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.session_state.loader = PyPDFLoader(temp_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    st.session_state.chunks = st.session_state.splitter.split_documents(st.session_state.docs)

    page_texts = [chunk.page_content for chunk in st.session_state.chunks]
    bm25_encoder.fit(page_texts)

    dense_vectors = embedding.embed_documents(page_texts)
    sparse_vectors = bm25_encoder.encode_documents(page_texts)

    pinecone_vectors = [
    Vector(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, st.session_state.chunks[i].page_content)),
        values=dense_vectors[i],
        sparse_values=sparse_vectors[i],
        metadata={
            **st.session_state.chunks[i].metadata,
            "context": st.session_state.chunks[i].page_content
        }
    )
    for i in range(len(page_texts))
]
    index.upsert(vectors=pinecone_vectors)

    from langchain_community.retrievers import PineconeHybridSearchRetriever
    st.session_state.retriever = PineconeHybridSearchRetriever(
        embeddings=embedding,
        sparse_encoder=bm25_encoder,
        index=index
    )
    st.success(f"Processed and stored {len(st.session_state.chunks)} chunks in Pinecone.")

button = st.button("Store Document in Pinecone DB")
query = st.text_input("Ask a question about the document")

if uploaded_file :
    if button:
        if "retriever" not in st.session_state:
            with st.spinner("Processing and storing document in Pinecone..."):
                vector_embedding()

    if query :
        with st.spinner("Searching and generating response..."):
            doc_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
            result = chain.invoke({"input": query})

            st.subheader("🔍 Response from Document")
            st.write(result["answer"])