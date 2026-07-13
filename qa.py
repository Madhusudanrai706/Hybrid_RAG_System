"""
QA module - prompt template, context building, and LLM invocation.

Uses functools.lru_cache instead of st.cache_resource so this file has
no Streamlit dependency - it works the same in console mode (main.py)
and inside the Streamlit app (app.py).
"""

import os
from functools import lru_cache

from langchain_groq import ChatGroq

from config import LLM_MODEL_NAME, LLM_TEMPERATURE

PROMPT_TEMPLATE = """You are an intelligent AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply exactly:
"I couldn't find this information in the uploaded PDFs."

Context:
{context}

Question:
{question}
"""


@lru_cache(maxsize=1)
def get_llm():
    """Loaded once per process, not on every call."""
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )


def build_context(top_docs):
    return "\n\n".join(doc.page_content for doc in top_docs)


def generate_answer(llm, context, question):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content