from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")


class MovieInfo(BaseModel):
    movie_title: str
    genre: List[str]
    release_year: str
    director: str
    main_cast: List[str]
    language: str
    country: str
    runtime: str
    plot_summary: str
    main_theme: str
    rating: str
    awards: List[str]
    box_office_collection: str
    streaming_platform: str
    important_keywords: List[str]


parser = PydanticOutputParser(pydantic_object=MovieInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent movie information extraction assistant.

Read the given movie paragraph carefully and extract useful information.

If information is missing, write "Not Mentioned".

{format_instructions}

Movie Paragraph:
{movie_paragraph}
""")
])

para = input("Give your paragraph: ")

final_prompt = prompt.invoke({
    "movie_paragraph": para,
    "format_instructions": parser.get_format_instructions()
})

response = model.invoke(final_prompt)

movie_data = parser.parse(response.content)

print("\nExtracted Movie Information:\n")

print(movie_data.model_dump_json(indent=4))
