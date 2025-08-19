import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Load .env file (local)
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")  # Local .env

# -----------------------------
# Fallback to Streamlit Secrets
# -----------------------------
if not api_key:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.sidebar.warning("Enter your GROQ API key in .env or Streamlit Secrets.")
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
# Initialize chat memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Assalam o Alaikum! I'm your Groq chatbot. How can I help you today?")
    ]

# -----------------------------
# Display chat messages
# -----------------------------
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

    # Prepare history for LLM
    history_for_llm = st.session_state.messages.copy()

    # Get model response
    with st.chat_message("assistant"):
        try:
            response = llm.invoke(history_for_llm)
            reply = response.content
            st.markdown(reply)
            st.session_state.messages.append(AIMessage(content=reply))
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("💻 Developed by **Laiba Rani**")
