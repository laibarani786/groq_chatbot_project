import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# --- Load .env for local ---
load_dotenv()

# --- API key handling (first Streamlit secrets, then .env) ---
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not api_key:
    st.error("🚨 API key not found! Please add it in Streamlit secrets or .env file.")
    st.stop()

# --- Initialize Chat Model ---
llm = ChatOpenAI(
    model_name="gemma2-9b-it",   # Groq ka model
    openai_api_key=api_key
)

# --- Conversation Memory ---
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Groq AI Chatbot")
st.caption("Powered by **LangChain + Groq** 🚀")
st.markdown("---")

# --- Chat history in session ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display previous messages ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Type your message..."):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                response = conversation.predict(input=prompt)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.markdown(response)

    # Save bot response
    st.session_state["messages"].append({"role": "assistant", "content": response})

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, LangChain & Groq API")
