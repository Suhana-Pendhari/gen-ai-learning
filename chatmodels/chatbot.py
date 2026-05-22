from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9)

messages = []

while True:
    prompt = input("You: ")
    messages.append(prompt)
    if(prompt == "0"):
        break
    response = model.invoke(messages)
    messages.append(response.content)
    print("Bot: ", response.content)
