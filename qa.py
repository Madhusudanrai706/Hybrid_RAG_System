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

Each context block below is labeled with its source file, page, and
chapter. When you use information from a block, mention which source
and page it came from.

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
    """
    Join the top retrieved chunks into one context string for the LLM.

    Each chunk is prefixed with a metadata header (source/page/chapter)
    so the LLM can cite where an answer came from - this matters more
    now that results can be metadata-filtered, so the person asking can
    see which specific PDF/page/chapter the answer is grounded in.
    """
    parts = []

    for doc in top_docs:
        meta = doc.metadata
        header = (
            f"[Source: {meta.get('source', 'Unknown')} | "
            f"Page: {meta.get('page', '-')} | "
            f"Chapter: {meta.get('chapter', 'Unknown')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(parts)


def generate_answer(llm, context, question):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content