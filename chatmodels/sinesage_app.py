import streamlit as st
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

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Information Extractor")
st.write("Extract structured movie information from paragraphs using AI.")

para = st.text_area(
    "Enter Movie Paragraph",
    height=250,
    placeholder="Paste movie paragraph here..."
)

if st.button("Extract Information"):

    if para.strip() == "":
        st.warning("Please enter a movie paragraph.")
    
    else:
        with st.spinner("Extracting information..."):

            final_prompt = prompt.invoke({
                "movie_paragraph": para,
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            movie_data = parser.parse(response.content)

            st.subheader("Extracted JSON Data")

            st.json(movie_data.model_dump())


