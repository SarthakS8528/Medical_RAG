from src.bm25_retriever import build_bm25_index, retrieve_bm25
from src.retriever import load_vector_store, retrieve_documents


DENSE_TOP_K = 20
BM25_TOP_K = 20

RRF_K = 60


def deduplicate_by_doc_id(results):
    """
    Convert chunk-level ranking into document-level ranking.

    Keeps the first occurrence of each document because
    the first occurrence represents the highest-ranked chunk
    retrieved for that document.
    """
    seen = set()
    unique_results = []

    for doc, score in results:
        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:
            seen.add(doc_id)
            unique_results.append((doc, score))

    return unique_results


def reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K):

    dense_results = deduplicate_by_doc_id(dense_results)
    bm25_results = deduplicate_by_doc_id(bm25_results)

    rrf_scores = {}

    # Dense ranking
    for rank, (doc, score) in enumerate(dense_results, start=1):

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "score": 0.0,
                "doc": doc
            }

        rrf_scores[doc_id]["score"] += 1 / (k + rank)

    # BM25 ranking
    for rank, (doc, score) in enumerate(bm25_results, start=1):

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "score": 0.0,
                "doc": doc
            }

        rrf_scores[doc_id]["score"] += 1 / (k + rank)

    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        (item["doc"], item["score"])
        for item in ranked_results
    ]
def retrieve_hybrid(
    vector_store,
    bm25,
    chunks,
    query,
    k=5
):
    """
    Retrieve documents using both Dense Retrieval
    and BM25, then combine their rankings using RRF.
    """

    dense_results = retrieve_documents(
        vector_store,
        query,
        k=DENSE_TOP_K
    )

    bm25_results = retrieve_bm25(
        bm25,
        chunks,
        query,
        k=BM25_TOP_K
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    return fused_results[:k]


def build_hybrid_retriever():

    print("Loading Dense vector store...")

    vector_store = load_vector_store()

    print("Dense vector store loaded.")

    print("\nBuilding BM25 index...")

    bm25, chunks = build_bm25_index()

    print("BM25 index built.")

    return vector_store, bm25, chunks


if __name__ == "__main__":

    vector_store, bm25, chunks = build_hybrid_retriever()

    query = (
        "What are the main biological problems that cause "
        "blood sugar regulation to break down in type 2 diabetes?"
    )

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    results = retrieve_hybrid(
        vector_store,
        bm25,
        chunks,
        query,
        k=5
    )

    print("\nTop Hybrid Results:\n")

    for rank, (doc, score) in enumerate(
        results,
        start=1
    ):

        print(f"Rank {rank}")
        print(f"RRF Score: {score:.6f}")
        print(
            f"Doc ID: "
            f"{doc.metadata.get('doc_id')}"
        )
        print(
            f"Text: "
            f"{doc.page_content[:300]}"
        )
        print("-" * 70)