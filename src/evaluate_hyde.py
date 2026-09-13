import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever import retrieve_documents,load_vector_store


DATASET_PATH = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"
HYDE_PATH = PROJECT_ROOT / "experiments" / "hyde_documents.json"


with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)


with open(HYDE_PATH, "r") as f:
    hyde_documents = json.load(f)


hyde_lookup = {
    item["id"]: item["hypothetical_document"]
    for item in hyde_documents
}


def deduplicate_documents(results):

    seen = set()
    unique_documents = []

    for doc, score in results:

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:

            seen.add(doc_id)
            unique_documents.append(doc)

    return unique_documents


def evaluate_retrieval(
    retrieved_docs,
    relevant_doc_ids,
    k=5
):

    retrieved_ids = [
        doc.metadata.get("doc_id")
        for doc in retrieved_docs[:k]
    ]

    relevant_doc_ids = set(relevant_doc_ids)

    # Hit@K
    hit = int(
        any(
            doc_id in relevant_doc_ids
            for doc_id in retrieved_ids
        )
    )

    # Recall@K
    retrieved_relevant = sum(
        1
        for doc_id in retrieved_ids
        if doc_id in relevant_doc_ids
    )

    recall = (
        retrieved_relevant / len(relevant_doc_ids)
        if relevant_doc_ids
        else 0
    )

    # Precision@K
    precision = (
        retrieved_relevant / len(retrieved_ids)
        if retrieved_ids
        else 0
    )

    # MRR@K
    reciprocal_rank = 0

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant_doc_ids:

            reciprocal_rank = 1 / rank
            break

    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank
    }


def main():

    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []

    recalls_at_5 = []
    precisions_at_5 = []
    mrrs = []
    vector_store = load_vector_store()
    print("\n" + "=" * 80)
    print("HYDE RETRIEVAL EVALUATION")
    print("=" * 80)

    for item in evaluation_dataset:

        question_id = item["id"]

        original_query = item["question"]

        relevant_doc_ids = item["relevant_doc_ids"]

        hypothetical_document = hyde_lookup.get(
            question_id
        )

        if hypothetical_document is None:

            print(
                f"\nQuestion {question_id}: "
                "HyDE document missing. Skipping."
            )

            continue

        # Retrieve a larger pool first
        retrieved_chunks = retrieve_documents(
            vector_store,
            hypothetical_document,
            k=20
        )

        # Deduplicate at document level
        retrieved_docs = deduplicate_documents(
            retrieved_chunks
        )

        # Keep Top-5 unique documents
        evaluation_docs = retrieved_docs[:5]

        metrics_1 = evaluate_retrieval(
            evaluation_docs,
            relevant_doc_ids,
            k=1
        )

        metrics_3 = evaluate_retrieval(
            evaluation_docs,
            relevant_doc_ids,
            k=3
        )

        metrics_5 = evaluate_retrieval(
            evaluation_docs,
            relevant_doc_ids,
            k=5
        )

        hits_at_1.append(metrics_1["hit"])
        hits_at_3.append(metrics_3["hit"])
        hits_at_5.append(metrics_5["hit"])

        recalls_at_5.append(
            metrics_5["recall"]
        )

        precisions_at_5.append(
            metrics_5["precision"]
        )

        mrrs.append(
            metrics_5["mrr"]
        )

        print(
            f"\nQ{question_id}: {original_query}"
        )

        print(
            f"Relevant: {relevant_doc_ids}"
        )

        print(
            "Retrieved:",
            [
                doc.metadata.get("doc_id")
                for doc in evaluation_docs
            ]
        )

        print(
            f"Hit@1: {metrics_1['hit']} | "
            f"Hit@3: {metrics_3['hit']} | "
            f"Hit@5: {metrics_5['hit']} | "
            f"Recall@5: {metrics_5['recall']:.3f} | "
            f"Precision@5: {metrics_5['precision']:.3f} | "
            f"MRR@5: {metrics_5['mrr']:.3f}"
        )

    n = len(hits_at_1)

    print("\n" + "=" * 80)
    print("FINAL HYDE RESULTS")
    print("=" * 80)

    print(
        f"Hit@1      : {sum(hits_at_1) / n:.4f}"
    )

    print(
        f"Hit@3      : {sum(hits_at_3) / n:.4f}"
    )

    print(
        f"Hit@5      : {sum(hits_at_5) / n:.4f}"
    )

    print(
        f"Recall@5   : {sum(recalls_at_5) / n:.4f}"
    )

    print(
        f"Precision@5: {sum(precisions_at_5) / n:.4f}"
    )

    print(
        f"MRR@5      : {sum(mrrs) / n:.4f}"
    )


if __name__ == "__main__":
    main()