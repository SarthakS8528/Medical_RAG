import os
from dotenv import load_dotenv
from google import genai



load_dotenv()

def get_gemini_client():
    api_key=os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Api key not found in .env")
    
    client=genai.Client(api_key=api_key)
    return client

def generate_answer(query, retrieved_documents):

    client = get_gemini_client()

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('doc_id', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in retrieved_documents
        ]
    )

    prompt=f"""
    You are a bio-medical question-asnwering assistant.
    
    Answer the user's question only based on the context provided.
    If the context does not contain enough information to answer the question,\
    say that the available context is insufficient.
    
    Do not introduce information that is not supported by the context at any cost.

    Context:
    {context}

    Question:
    {query}

    Answer

     """
    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return  response.text

