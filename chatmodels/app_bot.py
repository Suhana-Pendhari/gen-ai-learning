
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

bot_modes = {
    "😊 Happy Bot": "You are a very happy, cheerful and positive assistant.",
    "😡 Angry Bot": "You are an angry assistant who replies in a frustrated tone but never uses abusive language.",
    "😢 Sad Bot": "You are a sad and emotional assistant who replies in a depressed tone.",
    "🤖 Normal Bot": "You are a helpful AI assistant."
}

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")

selected_mode = st.selectbox(
    "Choose Bot Personality",
    list(bot_modes.keys())
)

system_prompt = bot_modes[selected_mode]

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
    chat_messages = [
        HumanMessage(content=system_prompt)
    ]

    for msg in st.session_state.messages:
        chat_messages.append(msg)

    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_messages.append(user_message)

        response = model.invoke(chat_messages)

        bot_reply = response.content

        st.markdown(bot_reply)

    st.session_state.messages.append(
        AIMessage(content=bot_reply)
    )

# streamlit run app_bot.py
