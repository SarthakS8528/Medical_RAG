from langchain_community.vectorstores import FAISS

from ingest import load_pubmed_documents
from chunking import chunk_documents
from embeddings import get_embedding_model


def build_vector_store():

    # 1. Load PubMed documents
    docs = load_pubmed_documents("pubmed_articles")

    # 2. Create chunks
    chunks = chunk_documents(
        docs,
        chunk_size=512,
        chunk_overlap=50
    )

    # 3. Load embedding model
    embedding_model = get_embedding_model()

    # 4. Create FAISS vector store
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


def retrieve_documents(vector_store, query, k=5):

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    vector_store = build_vector_store()

    query = "What is the role of metformin in DPP?"

    results = retrieve_documents(
        vector_store,
        query,
        k=5
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieved Documents:\n")

    for i, (doc, score) in enumerate(results):

        print(f"--- Result {i + 1} ---")
        print(f"Score: {score}")
        print(f"Doc ID: {doc.metadata.get('doc_id')}")
        print(f"Source: {doc.metadata.get('file_name')}")
        print(doc.page_content[:500])
        print()