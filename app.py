import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv('key.env')
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key not found. Please check your 'key.env' file.")
    st.stop()

client = OpenAI(api_key=api_key)


st.set_page_config(page_title="ElectroStore Assistant", page_icon="⚡")
st.title("⚡ ElectroStore Assistant")
st.markdown("Ask me anything about electronics, gadgets, components, or tech support.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful electronics store assistant for ElectroStore. You help customers with product information, technical support, and general electronics knowledge."}
    ]

for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::Bdw2s1cV",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
