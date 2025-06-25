# -------------------- Environment Setup --------------------
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "agents-test"
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

# -------------------- Wikipedia Tool --------------------
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

apiwrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=apiwrapper)

# -------------------- Web Loader + VectorStore --------------------
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectordb = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool

ret_tool = create_retriever_tool(
    retriever,
    "langchain_search",
    "Search any information about langchain, for any question about langchain tool, you should use this tool"
)



# -------------------- Arxiv Tool --------------------
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun

arxivwrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxivwrapper)

# -------------------- ToolKit --------------------
tools = [wiki_tool, arxiv_tool, ret_tool]

# -------------------- LLM + Prompt --------------------
from langchain import hub
from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

prompt = hub.pull("hwchase17/openai-functions-agent")

# -------------------- Agent Setup --------------------
from langchain.agents import create_openai_tools_agent

agent = create_openai_tools_agent(llm, tools, prompt)

# -------------------- Agent Executor --------------------
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

qs = input('Enter Your Query: ')
response = agent_executor.invoke({"input": qs})

print("|____________________________________________|")
print(response['output'])
