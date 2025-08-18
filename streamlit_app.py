"""
💬 Streamlit Chatbot using Groq (Gemma2 / LLaMA)
- Developed by Laiba Rani
- API key is loaded securely from .env or Streamlit Secrets
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.sidebar.warning("Enter your GROQ API key in .env file or Streamlit Secrets.")
    st.stop()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Groq Chatbot", page_icon="💬", layout="centered")
st.title("💬 Groq Chatbot")
st.caption("Developed by Laiba Rani — Minimal, fast, clean chat interface")

# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        ["gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.3, 0.1)

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name=model,
    temperature=temperature
)

# -----------------------------
# Chat memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Assalam o Alaikum! I'm your Groq chatbot. How can I help you today?")
    ]

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Type your message…")
if prompt:
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare history
    history_for_llm = []
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            history_for_llm.append(HumanMessage(content=m.content))
        else:
            history_for_llm.append(AIMessage(content=m.content))

    # Model response
    with st.chat_message("assistant"):
        try:
            response = llm.invoke(history_for_llm)
            reply = response.content
            st.markdown(reply)
            st.session_state.messages.append(AIMessage(content=reply))
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# Clear chat button
# -----------------------------
with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [AIMessage(content="Chat cleared. How can I help?")]
        st.experimental_rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("💻 Developed by **Laiba Rani**")
