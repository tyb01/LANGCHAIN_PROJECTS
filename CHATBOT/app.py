import os
import streamlit as st
import random
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import trim_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "llm_translator"

model = init_chat_model("llama3-8b-8192", model_provider="groq")

personalities = {
    "Pirate": "Talk like a fierce pirate. Use sea slang, arrr!",
    "Therapist": "Be a gentle and supportive therapist. Validate feelings.",
    "Sarcastic Teen": "Respond like a sarcastic and bored teenager.",
    "Ancient Monk": "Speak in proverbs and old wisdom like a Zen monk.",
    "Drill Sergeant": "Be loud, direct, and military-style bossy.",
    "Alien AI": "Talk like an AI from another galaxy, very formal and robotic.",
}

def get_prompt_template(personality_instruction):
    return ChatPromptTemplate.from_messages([
        ("system", personality_instruction),
        MessagesPlaceholder(variable_name="messages"),
    ])

trimmer = trim_messages(
    max_tokens=10000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    personality = random.choice(list(personalities.keys()))
    system_instruction = personalities[personality]
    prompt_template = get_prompt_template(system_instruction)

    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({
        "messages": trimmed_messages,
        "language": state["language"]
    })
    response = model.invoke(prompt)

    response.content = f"🧠 [{personality}]: {response.content}"
    return {"messages": [response]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==== Streamlit UI ====
st.set_page_config(page_title="🧠 Dr. MoodSwing", page_icon="🎭")
st.title("🎭 Dr. MoodSwing: The Unstable AI Therapist")

st.markdown("Every time you say something, Dr. MoodSwing responds with a **random personality.**\nExpect pirates, monks, teens, and aliens!")

# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("You:", key="user_input")


if  query:
    input_message = HumanMessage(content=query)
    config = {"configurable": {"thread_id": "mood_swing_001"}}
    language = "English"

    input_messages = st.session_state.chat_history + [input_message]

    output = app.invoke(
        {"messages": input_messages, "language": language},
        config
    )

    response = output["messages"][-1]
    st.session_state.chat_history += [input_message, response]


if query:
    st.markdown(f"🧍‍♂️ **You:** {query}")
    st.markdown(f"🤖 {response.content}")

