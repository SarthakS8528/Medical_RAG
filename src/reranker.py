from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    """
    Load the Cross-Encoder reranking model.
    """
    model = CrossEncoder(MODEL_NAME)
    return model


def rerank_documents(model, query, documents, top_k=5):
    """
    Rerank documents using a Cross-Encoder and return
    the top-k unique documents.
    """

    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = model.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Deduplicate by doc_id
    seen = set()
    unique_documents = []

    for doc, score in ranked_documents:

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:
            seen.add(doc_id)
            unique_documents.append((doc, score))

        if len(unique_documents) == top_k:
            break

    return unique_documents