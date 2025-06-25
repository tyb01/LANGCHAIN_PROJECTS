import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox


from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("OBJECTBOS VECTORSTOREDB WITH GROQ MODEL")

llm = ChatGroq(model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the quesiton based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question : {input}
"""
)

#vector embedding

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-mpnet-base-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = ObjectBox.from_documents(
            documents=st.session_state.final_docs,
            embedding=st.session_state.embeddings,
            embedding_dimension =  768
        )


input_prompt = st.text_input("Enter your question from documents")
if st.button("Document Embeddings"):
    vector_embedding()
    st.write("ObjectBox database is ready")
if input_prompt:

    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()

    retriever_chain = create_retrieval_chain(retriever,document_chain)

    response = retriever_chain.invoke(input_prompt)

    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------")