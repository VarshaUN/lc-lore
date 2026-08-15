import os
import re
import json
import streamlit as st
from PIL import Image
import pytesseract

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_community.tools.tavily_search import TavilySearchResults


st.set_page_config(page_title="Personal Chef Agent", page_icon="🍳", layout="centered")
st.title("🍳 Personal Chef Agent")
st.write("Enter leftover ingredients or upload a photo, and I’ll suggest recipes.")


def clean_ingredients(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9,\s\-]", " ", text)
    parts = [x.strip() for x in text.split(",")]
    parts = [x for x in parts if x]
    return parts


@tool
def recipe_search(query: str) -> str:
    """Search the web for recipe ideas based on ingredients."""
    search = TavilySearchResults(max_results=5)
    results = search.invoke({"query": query})
    if isinstance(results, list):
        return "\n\n".join(
            [f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}"
             for r in results]
        )
    return str(results)


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful personal chef agent. "
     "Given leftover ingredients, suggest 3 easy recipes. "
     "Use web search results if available. "
     "For each recipe include: title, why it fits, ingredients used, and quick steps. "
     "If ingredients are unclear from image OCR, mention that politely and make the best guess."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
tools = [recipe_search]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


user_text = st.text_area("Type leftover ingredients (comma separated)", placeholder="rice, onion, tomato, egg, bread")
uploaded_image = st.file_uploader("Or upload an image of leftovers", type=["png", "jpg", "jpeg"])

final_ingredients = ""

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded leftovers", use_container_width=True)
    extracted_text = extract_text_from_image(uploaded_image)
    st.subheader("Extracted text from image")
    st.write(extracted_text if extracted_text.strip() else "No clear text found.")
    final_ingredients = extracted_text.strip()

if user_text.strip():
    final_ingredients = user_text.strip()

if st.button("Suggest Recipes"):
    if not final_ingredients:
        st.warning("Please enter ingredients or upload an image.")
    else:
        ingredients_list = clean_ingredients(final_ingredients)
        query = (
            f"Find recipe ideas for these leftover ingredients: {', '.join(ingredients_list)}. "
            f"Suggest practical recipes and common substitutions."
        )
        response = executor.invoke({"input": query})
        st.subheader("Recipe Suggestions")
        st.write(response["output"])