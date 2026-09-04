from hybrid_retriever import build_hybrid_retriever, retrieve_hybrid
from hybrid_retriever import build_hybrid_retriever, retrieve_hybrid

import json
from pathlib import Path


# Load evaluation dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"

with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)

with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)

print("\nEvaluation dataset structure:")
print(evaluation_dataset[0])


def get_unique_doc_ids(results):
    """
    Extract unique document IDs while preserving retrieval order.
    """
    seen = set()
    unique_docs = []

    for doc, score in results:
        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc_id)

    return unique_docs


def evaluate_hybrid(vector_store, bm25, chunks):

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    recall_at_5 = 0
    precision_at_5 = 0
    mrr = 0

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 80)

    for i, item in enumerate(evaluation_dataset, start=1):

        query = item["question"]
        relevant_docs = set(item["relevant_doc_ids"])

        results = retrieve_hybrid(
            vector_store,
            bm25,
            chunks,
            query,
            k=20
        )

        retrieved_docs = get_unique_doc_ids(results)

        top_1 = retrieved_docs[:1]
        top_3 = retrieved_docs[:3]
        top_5 = retrieved_docs[:5]

        # -------------------------
        # Hit@K
        # -------------------------

        if relevant_docs.intersection(top_1):
            hit_at_1 += 1

        if relevant_docs.intersection(top_3):
            hit_at_3 += 1

        if relevant_docs.intersection(top_5):
            hit_at_5 += 1

        # -------------------------
        # Recall@5
        # -------------------------

        relevant_retrieved = relevant_docs.intersection(top_5)

        recall_at_5 += (
            len(relevant_retrieved) / len(relevant_docs)
        )

        # -------------------------
        # Precision@5
        # -------------------------

        precision_at_5 += (
            len(relevant_retrieved) / len(top_5)
        )

        # -------------------------
        # MRR
        # -------------------------

        reciprocal_rank = 0

        for rank, doc_id in enumerate(retrieved_docs, start=1):

            if doc_id in relevant_docs:
                reciprocal_rank = 1 / rank
                break

        mrr += reciprocal_rank

        # -------------------------
        # Print question-level result
        # -------------------------

        print(f"\nQ{i}: {query}")

        print(f"Relevant Docs : {sorted(relevant_docs)}")
        print(f"Retrieved Top 5: {top_5}")

        print(
            f"Hit@1={int(bool(set(top_1) & relevant_docs))} | "
            f"Hit@3={int(bool(set(top_3) & relevant_docs))} | "
            f"Hit@5={int(bool(set(top_5) & relevant_docs))} | "
            f"Recall@5={len(relevant_retrieved) / len(relevant_docs):.3f} | "
            f"MRR={reciprocal_rank:.3f}"
        )

    # -------------------------
    # Average metrics
    # -------------------------

    n = len(evaluation_dataset)

    print("\n" + "=" * 80)
    print("FINAL HYBRID RESULTS")
    print("=" * 80)

    print(f"Hit@1      : {hit_at_1 / n:.4f}")
    print(f"Hit@3      : {hit_at_3 / n:.4f}")
    print(f"Hit@5      : {hit_at_5 / n:.4f}")
    print(f"Recall@5   : {recall_at_5 / n:.4f}")
    print(f"Precision@5: {precision_at_5 / n:.4f}")
    print(f"MRR        : {mrr / n:.4f}")


if __name__ == "__main__":

    vector_store, bm25, chunks = build_hybrid_retriever()

    evaluate_hybrid(
        vector_store,
        bm25,
        chunks
    )