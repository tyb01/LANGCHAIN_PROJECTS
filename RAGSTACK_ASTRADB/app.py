from dotenv import load_dotenv
import os
load_dotenv()


import numpy
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name = model_name)

vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embeddings,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)
print("Astra vector store configured")

from datasets import load_dataset

dt = load_dataset("datastax/philosopher-quotes")['train']


from langchain.schema import Document

docs = []
for entry in dt:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)


vstore.add_documents(docs)

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retriever = vstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = init_chat_model("llama3-8b-8192", model_provider="groq")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("In the given context, what is the most important to allow the brain and provide me the tags?")