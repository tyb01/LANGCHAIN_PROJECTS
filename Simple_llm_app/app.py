import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "llm_translator"

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

system_template = "Translate the following from English into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

output_parser=StrOutputParser()
chain=prompt_template|llm|output_parser

st.set_page_config(page_title="LLM Translator", page_icon="🌐")
st.title("🌐 LLM Translator App")


text = st.text_area("Enter English text to translate:")
language = st.text_input("Target language (e.g., Spanish, French, Urdu):")

# Translate button
if st.button("Translate"):
    if not text or not language:
        st.warning("Please provide both the text and the target language.")
    else:
        try:
            
            response = chain.invoke({"language": language, "text": text})
            st.write(response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
