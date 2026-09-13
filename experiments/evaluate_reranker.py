import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever import load_vector_store, retrieve_documents
from src.reranker import load_reranker, rerank_documents


DATASET_PATH = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"


def deduplicate_documents(results):
    """
    Convert ranked (Document, score) results into
    unique documents while preserving ranking.
    """

    seen = set()
    unique_documents = []

    for doc, score in results:

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:
            seen.add(doc_id)
            unique_documents.append(doc)

    return unique_documents


def evaluate_retrieval(retrieved_docs, relevant_doc_ids, k=5):

    retrieved_ids = [
        doc.metadata.get("doc_id")
        for doc in retrieved_docs[:k]
    ]

    relevant_doc_ids = set(relevant_doc_ids)

    # Hit@K
    hit = int(
        any(doc_id in relevant_doc_ids for doc_id in retrieved_ids)
    )

    # Recall@K
    relevant_retrieved = sum(
        doc_id in relevant_doc_ids
        for doc_id in retrieved_ids
    )

    recall = (
        relevant_retrieved / len(relevant_doc_ids)
        if relevant_doc_ids
        else 0
    )

    # Precision@K
    precision = (
        relevant_retrieved / len(retrieved_ids)
        if retrieved_ids
        else 0
    )

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in relevant_doc_ids:
            reciprocal_rank = 1 / rank
            break

    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank
    }


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading vector store...")
vector_store = load_vector_store()

print("Loading Cross-Encoder...")
reranker = load_reranker()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

hit_at_1 = []
hit_at_3 = []
hit_at_5 = []

recall_at_5 = []
precision_at_5 = []
mrr_at_5 = []


for item in evaluation_dataset:

    question_id = item["id"]
    query = item["question"]
    relevant_doc_ids = item["relevant_doc_ids"]

    print(f"\nQuestion {question_id}/30")
    print(f"Query: {query}")
    print(f"Relevant: {relevant_doc_ids}")

    # ----------------------------------------------
    # Stage 1: Dense Retrieval
    # ----------------------------------------------

    dense_results = retrieve_documents(
        vector_store,
        query,
        k=20
    )

    # ----------------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # ----------------------------------------------

    candidate_documents = [
        doc
        for doc, score in dense_results
    ]

    reranked_results = rerank_documents(
        reranker,
        query,
        candidate_documents,
        top_k=20
    )

    # rerank_documents returns (Document, score)
    reranked_documents = deduplicate_documents(
        reranked_results
    )

    # ----------------------------------------------
    # Evaluate Top-5
    # ----------------------------------------------

    evaluation_docs = reranked_documents[:5]

    metrics = evaluate_retrieval(
        evaluation_docs,
        relevant_doc_ids,
        k=5
    )

    hit_at_1.append(
        int(
            any(
                doc.metadata.get("doc_id") in set(relevant_doc_ids)
                for doc in evaluation_docs[:1]
            )
        )
    )

    hit_at_3.append(
        int(
            any(
                doc.metadata.get("doc_id") in set(relevant_doc_ids)
                for doc in evaluation_docs[:3]
            )
        )
    )

    hit_at_5.append(metrics["hit"])

    recall_at_5.append(metrics["recall"])
    precision_at_5.append(metrics["precision"])
    mrr_at_5.append(metrics["reciprocal_rank"])

    print("Reranked Top-5:")

    for rank, doc in enumerate(evaluation_docs, start=1):
        print(
            f"  {rank}. "
            f"{doc.metadata.get('doc_id')}"
        )


# --------------------------------------------------
# Final Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("CROSS-ENCODER RERANKING RESULTS")
print("=" * 50)

print(
    f"Hit@1       : {sum(hit_at_1) / len(hit_at_1):.4f}"
)

print(
    f"Hit@3       : {sum(hit_at_3) / len(hit_at_3):.4f}"
)

print(
    f"Hit@5       : {sum(hit_at_5) / len(hit_at_5):.4f}"
)

print(
    f"Recall@5    : {sum(recall_at_5) / len(recall_at_5):.4f}"
)

print(
    f"Precision@5 : {sum(precision_at_5) / len(precision_at_5):.4f}"
)

print(
    f"MRR@5       : {sum(mrr_at_5) / len(mrr_at_5):.4f}"
)