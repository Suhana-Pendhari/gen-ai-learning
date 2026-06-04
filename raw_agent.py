#first step - loading all the libraries
from dotenv import load_dotenv

load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

#Now lets create some tools

#weather tool
@tool
def get_weather(city: str) -> str:
    """Get Current weather of a City"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    print("DEBUG:", data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f"Weather in {city}: {desc}, {temp}°C"


#tavily news tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
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
            f"- {title}\n 🔗 {url}\n 📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n"+"\n\n".join(news_list)

print(get_news.invoke("Sangli"))

llm = ChatMistralAI(model="mistral-small-2506")

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tool = llm.bind_tools([get_weather, get_news])

#Agent LOOP - Very important

messages = []

print("City intelligence")
print("type Exit to quite")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)

        messages.append(result)

        #if tool is required

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                #human in the loop
                confirm = input("Agent wants to call {tool_name} Approve (yes/no)")

                if confirm.lower()!="yes":
                    print("tool call denied and I cannot et the latest information ")
                    break

                #execute tool
                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))

            continue

        else:
            print(result.content)


