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
    Rerank documents according to their relevance to the query.

    Parameters:
        model: loaded CrossEncoder
        query: original user query
        documents: list of LangChain Document objects
        top_k: number of documents to return

    Returns:
        List of (Document, score) tuples sorted by descending score.
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

    return ranked_documents[:top_k]