from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

st.set_page_config(
    page_title="Mistral Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Mistral AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Type your message...")

if prompt:
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = model.invoke(st.session_state.messages)

        bot_reply = response.content

        st.markdown(bot_reply)

    st.session_state.messages.append(
        AIMessage(content=bot_reply)
    )
