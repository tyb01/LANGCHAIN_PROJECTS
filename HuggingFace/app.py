from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
import os

# # Load documents
# loader = PyPDFDirectoryLoader("./pdfs")
# documents = loader.load()

# # Text splitting
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# final_docs = text_splitter.split_documents(documents)

# model_name = "sentence-transformers/all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(model_name=model_name)


# vector_store = FAISS.from_documents(
#     documents=final_docs,
#     embedding=embeddings
# )

# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3}
# )


os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
print(llm.invoke("What is hugging face"))

# prompt_template = """
# Use the following context to answer the question. 
# Provide answers ONLY from the context. If unsure, say "I don't know".

# Context:
# {context}

# Question: {input} 

# Answer:"""

# prompt = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "input"]  
# )

# document_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, document_chain)

# query = "What are major heart diseases"
# result = chain.invoke({"input": query})  
# print(result["answer"])
