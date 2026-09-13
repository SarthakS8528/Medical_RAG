import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API key not found in .env")

    return genai.Client(api_key=api_key)


def generate_hypothetical_document(query):
    client = get_gemini_client()

    prompt = f"""
You are a biomedical information retrieval assistant.

Given the user's biomedical question, write a short hypothetical
answer that resembles the kind of information that might appear
in a relevant PubMed biomedical abstract.

Rules:
1. Focus directly on the user's question.
2. Use biomedical terminology that is likely to appear in research literature.
3. Include relevant mechanisms, biological processes, conditions,
   or relationships implied by the question.
4. Do not mention that the answer is hypothetical.
5. Do not provide citations or references.
6. Do not introduce unrelated topics.
7. Keep the response concise and information-dense.
8. Do not use bullet points or headings.
9. Return only the hypothetical biomedical answer.

User question:
{query}

Hypothetical biomedical answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()