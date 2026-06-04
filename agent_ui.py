from dotenv import load_dotenv

load_dotenv()

import os
import requests
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="City Intelligence",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 City Intelligence Agent")
st.markdown("Ask about weather, news, or anything related to a city.")


# =========================
# TOOLS
# =========================

@tool
def get_weather(city: str) -> str:
    """Get Current weather of a City"""

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"🌤 Weather in {city}: {desc}, {temp}°C"


tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"### {title}\n"
            f"🔗 {url}\n\n"
            f"{snippet[:150]}..."
        )

    return "\n\n---\n\n".join(news_list)


# =========================
# LLM SETUP
# =========================

llm = ChatMistralAI(
    model="mistral-small-2506"
)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools(
    [get_weather, get_news]
)


# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("⚙️ Settings")

    auto_approve = st.toggle(
        "Auto Approve Tool Calls",
        value=True
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.langchain_messages = []
        st.rerun()


# =========================
# DISPLAY CHAT HISTORY
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# CHAT INPUT
# =========================

if prompt := st.chat_input("Ask something about a city..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.session_state.langchain_messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        while True:

            result = llm_with_tools.invoke(
                st.session_state.langchain_messages
            )

            st.session_state.langchain_messages.append(result)

            # =========================
            # TOOL CALLS
            # =========================

            if result.tool_calls:

                for tool_call in result.tool_calls:

                    tool_name = tool_call["name"]

                    st.info(
                        f"🔧 Calling Tool: {tool_name}"
                    )

                    if not auto_approve:
                        st.warning(
                            f"Tool '{tool_name}' requires approval."
                        )
                        break

                    tool_result = tools[tool_name].invoke(
                        tool_call
                    )

                    with st.expander(
                        f"Tool Output - {tool_name}",
                        expanded=True
                    ):
                        st.markdown(tool_result)

                    st.session_state.langchain_messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        )
                    )

                continue

            # =========================
            # FINAL RESPONSE
            # =========================

            else:

                response_placeholder.markdown(
                    result.content
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.content
                    }
                )

                break
            