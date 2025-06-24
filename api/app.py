from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import os
import uvicorn
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# === Load env vars ===
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "api-test"

# === FastAPI Setup ===
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server",
    openapi_url="/openapi.json",
    docs_url="/docs"
)

model1 = init_chat_model("llama3-8b-8192", model_provider="groq")
model2 = init_chat_model("llama3-8b-8192", model_provider="groq")

pr1 = ChatPromptTemplate.from_template(
    "Write a funny essay about {topic} of 300 words in a very decent way"
)
pr2 = ChatPromptTemplate.from_template(
    "Write a funny poem about {topic} of 300 words in a very decent way"
)

class EssayRequest(BaseModel):
    topic: str

@app.post("/essay")
async def generate_essay(request: EssayRequest):
    prompt = pr1.invoke({"topic": request.topic})
    response = model1.invoke(prompt)
    return JSONResponse(content={"essay": response.content})

@app.post("/poem")
async def generate_poem(request: EssayRequest):
    prompt = pr2.invoke({"topic": request.topic})
    response = model2.invoke(prompt)
    return JSONResponse(content={"poem": response.content})

@app.get("/openai")
async def openai_status():
    return {"message": "OpenAI route placeholder. Use essay/poem routes."}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=1000
                )
