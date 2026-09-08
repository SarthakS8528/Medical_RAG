# This will load Gemini API
# Take the user query
# Ask gemini to expand it under strinct prompot 
# Return that expanded query
# Send that expanded query to our existing Dense Retriever.


import os
from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API key not found in .env")

    client = genai.Client(api_key=api_key)

    return client


def expand_query(query):
    """
    Expand a user's biomedical question into a richer
    retrieval-oriented query.

    The model should NOT answer the question.
    It should only rewrite/expand it for retrieval.
    """

    client = get_gemini_client()

    prompt = f"""
You are a biomedical information retrieval assistant.

Your task is to expand the user's question into a richer
search query that will improve retrieval from a collection
of PubMed biomedical abstracts.

Rules:
1. Preserve the original intent of the question.
2. Add useful biomedical terminology, synonyms, and related
   concepts when appropriate.
3. Do not answer the question.
4. Do not introduce specific facts, findings, treatments,
   diseases, or mechanisms that are not implied by the
   original question.
5. Do not change the scope of the question.
6. Return ONLY the expanded query.
7. Do not include explanations, labels, or quotation marks.
8. Do not make the query unecessarily long.
9.Return ONLY one natural-language expanded query

Original question:
{query}

Expanded query:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()