from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent movie information extraction assistant.

Read the given movie paragraph carefully and extract useful information from it.

Present the extracted details in a clean structured format using headings.

Rules:
1. Do not return JSON.
2. Use proper labels and formatting.
3. If information is missing, write "Not Mentioned".
4. Keep the plot summary short and clear.
5. Extract only information present in the paragraph.
6. Do not add extra explanations.

Movie Paragraph:
{movie_paragraph}
""")
])

para = input("Give your paragraph: ")
final_prompt = prompt.invoke(
    {"movie_paragraph": para}
)

response = model.invoke(final_prompt)
print(response.content)
