import os

from langchain_community.vectorstores import FAISS

from ingest import load_pubmed_documents
from chunking import chunk_documents
from embeddings import get_embedding_model


VECTOR_STORE_PATH = "vector_store/faiss_index"


def build_vector_store():
    """
    Build a FAISS vector store from the PubMed documents.
    """

    docs = load_pubmed_documents("pubmed_articles")

    chunks = chunk_documents(
        docs,
        chunk_size=512,
        chunk_overlap=50
    )

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


def save_vector_store(vector_store):
    """
    Save the FAISS vector store locally.
    """

    os.makedirs("vector_store", exist_ok=True)

    vector_store.save_local(VECTOR_STORE_PATH)

    print("FAISS index saved successfully.")


def load_vector_store():
    """
    Load the previously saved FAISS vector store.
    """

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_documents(vector_store, query, k=5):
    """
    Retrieve the top-k most similar chunks for a query.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    

    vector_store = build_vector_store()

    save_vector_store(vector_store)

    print("Vector store is ready.")