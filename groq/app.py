import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

##loading apis
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "groq-test"

groq_api_key = os.getenv('GROQ_API_KEY')
model_name = "sentence-transformers/all-mpnet-base-v2"

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    st.session_state.loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size  = 1000, chunk_overlap = 20)
    st.session_state.chunk_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)


    st.session_state.vectors = FAISS.from_documents(
        st.session_state.chunk_documents,
        st.session_state.embeddings
    )


st.title("ChatGroq Demo")

llm = ChatGroq(groq_api_key= groq_api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")

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


document_chain = create_stuff_documents_chain(llm,prompt)

retriever = st.session_state.vectors.as_retriever()

retriever_chain = create_retrieval_chain(retriever,document_chain)


input=st.text_input("Enter your question here")


if input:
    start = time.process_time()
    response = retriever_chain.invoke({"input":input})
    end = time.process_time()

    print("Response time : ",end-start)

    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        #finding the relevant chunks

        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------")
