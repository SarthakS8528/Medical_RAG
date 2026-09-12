import json
import sys
from pathlib import Path

# ------------------------------------------------
# Add project root to Python path
# ------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever import load_vector_store, retrieve_documents


EVALUATION_DATASET = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"
EXPANSIONS_FILE = PROJECT_ROOT / "experiments" / "query_expansions.json"

RETRIEVAL_POOL_SIZE = 20
EVALUATION_K = 5


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deduplicate_doc_ids(retrieved_doc_ids):

    unique_doc_ids = []
    seen = set()

    for doc_id in retrieved_doc_ids:

        if doc_id not in seen:

            unique_doc_ids.append(doc_id)
            seen.add(doc_id)

    return unique_doc_ids


def hit_at_k(retrieved_doc_ids, relevant_doc_ids, k):

    retrieved_top_k = retrieved_doc_ids[:k]

    return float(
        any(
            doc_id in relevant_doc_ids
            for doc_id in retrieved_top_k
        )
    )


def recall_at_k(retrieved_doc_ids, relevant_doc_ids, k):

    retrieved_top_k = retrieved_doc_ids[:k]

    relevant_retrieved = sum(
        doc_id in relevant_doc_ids
        for doc_id in retrieved_top_k
    )

    if not relevant_doc_ids:
        return 0.0

    return relevant_retrieved / len(relevant_doc_ids)


def precision_at_k(retrieved_doc_ids, relevant_doc_ids, k):

    retrieved_top_k = retrieved_doc_ids[:k]

    if not retrieved_top_k:
        return 0.0

    relevant_retrieved = sum(
        doc_id in relevant_doc_ids
        for doc_id in retrieved_top_k
    )

    return relevant_retrieved / len(retrieved_top_k)


def reciprocal_rank(retrieved_doc_ids, relevant_doc_ids):

    for rank, doc_id in enumerate(
        retrieved_doc_ids,
        start=1
    ):

        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_doc_ids,
    relevant_doc_ids,
    k=EVALUATION_K
):

    return {

        "Hit@1": hit_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            1
        ),

        "Hit@3": hit_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            3
        ),

        "Hit@5": hit_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            5
        ),

        "Recall@5": recall_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            k
        ),

        "Precision@5": precision_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            k
        ),

        "MRR": reciprocal_rank(
            retrieved_doc_ids,
            relevant_doc_ids
        )
    }


def evaluate_dataset(
    dataset,
    expansions,
    vector_store,
    retrieval_pool_size=RETRIEVAL_POOL_SIZE,
    evaluation_k=EVALUATION_K
):

    # Create lookup table:
    #
    # question ID → expanded query
    #
    expansion_lookup = {
        item["id"]: item["expanded_query"]
        for item in expansions
    }

    all_results = []

    for i, item in enumerate(dataset):

        question_id = item["id"]

        original_query = item["question"]

        relevant_doc_ids = item["relevant_doc_ids"]

        expanded_query = expansion_lookup[question_id]

        # ------------------------------------------------
        # Step 1: Retrieve using EXPANDED query
        # ------------------------------------------------

        results = retrieve_documents(
            vector_store,
            expanded_query,
            k=retrieval_pool_size
        )

        # ------------------------------------------------
        # Step 2: Convert chunks → document IDs
        # ------------------------------------------------

        retrieved_doc_ids = [
            doc.metadata.get("doc_id")
            for doc, score in results
        ]

        # ------------------------------------------------
        # Step 3: Deduplicate documents
        # ------------------------------------------------

        unique_retrieved_doc_ids = deduplicate_doc_ids(
            retrieved_doc_ids
        )

        # ------------------------------------------------
        # Step 4: Take top-K unique documents
        # ------------------------------------------------

        evaluation_docs = unique_retrieved_doc_ids[
            :evaluation_k
        ]

        # ------------------------------------------------
        # Step 5: Calculate metrics
        # ------------------------------------------------

        metrics = evaluate_retrieval(
            evaluation_docs,
            relevant_doc_ids,
            k=evaluation_k
        )

        all_results.append({

            "id": question_id,

            "original_query": original_query,

            "expanded_query": expanded_query,

            "relevant_doc_ids": relevant_doc_ids,

            "retrieved_doc_ids": evaluation_docs,

            "metrics": metrics
        })

        print(
            f"[{i + 1}/{len(dataset)}] "
            f"Q{question_id}"
        )

        print(
            "Original:",
            original_query
        )

        print(
            "Expanded:",
            expanded_query
        )

        print(
            "Retrieved:",
            evaluation_docs
        )

        print(
            "Metrics:",
            {
                key: round(value, 3)
                for key, value in metrics.items()
            }
        )

        print()

    return all_results


def calculate_average_metrics(results):

    metric_names = [

        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Recall@5",
        "Precision@5",
        "MRR"
    ]

    averages = {}

    for metric in metric_names:

        values = [
            result["metrics"][metric]
            for result in results
        ]

        averages[metric] = (
            sum(values) / len(values)
        )

    return averages


if __name__ == "__main__":

    print(
        "Loading evaluation dataset..."
    )

    dataset = load_json(
        EVALUATION_DATASET
    )

    print(
        f"Loaded {len(dataset)} evaluation questions."
    )

    print(
        "\nLoading frozen query expansions..."
    )

    expansions = load_json(
        EXPANSIONS_FILE
    )

    print(
        f"Loaded {len(expansions)} expanded queries."
    )

    print(
        "\nLoading FAISS vector store..."
    )

    vector_store = load_vector_store()

    print(
        "Vector store loaded successfully.\n"
    )

    print("=" * 70)

    print(
        "RUNNING DENSE + QUERY EXPANSION EVALUATION"
    )

    print("=" * 70)

    print()

    results = evaluate_dataset(
        dataset,
        expansions,
        vector_store,
        retrieval_pool_size=RETRIEVAL_POOL_SIZE,
        evaluation_k=EVALUATION_K
    )

    averages = calculate_average_metrics(
        results
    )

    print("=" * 70)

    print(
        "QUERY EXPANSION RESULTS"
    )

    print("=" * 70)

    for metric, value in averages.items():

        print(
            f"{metric}: {value:.4f}"
        )