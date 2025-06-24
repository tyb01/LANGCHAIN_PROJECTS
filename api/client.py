import requests
import streamlit as st

# 🔗 Set your backend API base URL here (change if deployed elsewhere)
BASE_URL = "http://localhost:1000"

# === Request Essay ===
def essay_response(input_text):
    try:
        response = requests.post(
            f"{BASE_URL}/essay",
            json={"topic": input_text}
        )
        response.raise_for_status()  # Raise error if response code isn't 200
        return response.json().get("essay", "No essay found.")
    except Exception as e:
        return f"❌ Error: {e}"

# === Request Poem ===
def poem_response(input_text):
    try:
        response = requests.post(
            f"{BASE_URL}/poem",
            json={"topic": input_text}
        )
        response.raise_for_status()
        return response.json().get("poem", "No poem found.")
    except Exception as e:
        return f"❌ Error: {e}"

# === Streamlit UI ===
st.set_page_config(page_title="LangChain Essay & Poem Generator", page_icon="📝")
st.title("📝 LangChain Essay & Poem Generator (via API)")

st.markdown("Generate a funny **essay** or **poem** using the power of LLaMA 3 through a backend API.")

# === Essay Input ===
input1 = st.text_input('📘 Enter topic for an essay:')
if input1:
    st.subheader("✍️ Essay Output")
    st.write(essay_response(input1))

# === Poem Input ===
input2 = st.text_input('📜 Enter topic for a poem:')
if input2:
    st.subheader("🎨 Poem Output")
    st.write(poem_response(input2))
