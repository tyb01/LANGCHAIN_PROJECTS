import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

st.set_page_config(page_title="PDF Folder Q&A")
st.title("Gemma PDF Folder Q&A")

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")

prompt =ChatPromptTemplate.from_template(
"""
Answer the quesiton based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question : {input}
"""
)
print(3)
folder_path = st.text_input("Enter path to folder containing PDFs (Full path to pdfs)")

if st.button("Load & Create Vector Store"):
    if not os.path.isdir(folder_path):
        st.error("❌ Invalid folder path.")
    else:
        with st.spinner("🔄 Processing documents..."):
            loader = PyPDFDirectoryLoader(folder_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.retriever = vectorstore.as_retriever()
            st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.qa_chain = create_retrieval_chain(
                st.session_state.retriever,
                st.session_state.document_chain
            )

        st.success("✅ Vector store is ready!")

qs = st.text_input("Ask a question from the documents")

if qs and "qa_chain" in st.session_state:
    with st.spinner("Getting answer..."):
        response = st.session_state.qa_chain.invoke({"input": qs})
        st.success(response["answer"])

        with st.expander("Source Context"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("-----------------------")
elif qs:
    st.warning("⚠️ Please load documents first by providing a valid folder.")
