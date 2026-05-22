from dotenv import load_dotenv
load_dotenv()

import tkinter as tk
from tkinter import scrolledtext

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

messages = []

def send_message():
    user_input = entry_box.get()

    if user_input.strip() == "":
        return

    chat_area.config(state=tk.NORMAL)

    chat_area.insert(tk.END, f"You: {user_input}\n\n")

    messages.append(HumanMessage(content=user_input))

    entry_box.delete(0, tk.END)

    response = model.invoke(messages)

    bot_reply = response.content

    messages.append(AIMessage(content=bot_reply))

    chat_area.insert(tk.END, f"Bot: {bot_reply}\n\n")

    chat_area.config(state=tk.DISABLED)

    chat_area.yview(tk.END)

root = tk.Tk()
root.title("Mistral AI Chatbot")
root.geometry("600x700")
root.configure(bg="#1e1e1e")

title = tk.Label(
    root,
    text="Mistral AI Chatbot",
    font=("Arial", 18, "bold"),
    bg="#1e1e1e",
    fg="white"
)
title.pack(pady=10)

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 12),
    bg="#2b2b2b",
    fg="white",
    insertbackground="white"
)

chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_area.config(state=tk.DISABLED)

bottom_frame = tk.Frame(root, bg="#1e1e1e")
bottom_frame.pack(fill=tk.X, padx=10, pady=10)

entry_box = tk.Entry(
    bottom_frame,
    font=("Arial", 13),
    bg="#2b2b2b",
    fg="white",
    insertbackground="white"
)

entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=8)

send_button = tk.Button(
    bottom_frame,
    text="Send",
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    command=send_message
)

send_button.pack(side=tk.RIGHT)

root.bind('<Return>', lambda event: send_message())

root.mainloop()
