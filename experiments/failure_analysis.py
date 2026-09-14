import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


from src.retriever import load_vector_store, retrieve_documents

from src.bm25_retriever import (
    build_bm25_index,
    retrieve_bm25
)

from src.hybrid_retriever import (
    retrieve_hybrid,
    reciprocal_rank_fusion
)

from src.reranker import (
    load_reranker,
    rerank_documents
)

from src.evaluation import (
    get_unique_doc_ranking,
    get_relevant_ranks,
    classify_failure
)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "evaluation_dataset.json"
)

QUERY_EXPANSION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "query_expansions.json"
)

HYDE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "hyde_documents.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "failure_analysis_results.json"
)


# =========================================================
# LOAD DATASETS
# =========================================================

with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)

with open(QUERY_EXPANSION_PATH, "r") as f:
    query_expansions = json.load(f)

with open(HYDE_PATH, "r") as f:
    hyde_documents = json.load(f)


query_expansion_map = {
    item["id"]: item["expanded_query"]
    for item in query_expansions
}

hyde_map = {
    item["id"]: item["hypothetical_document"]
    for item in hyde_documents
}


# =========================================================
# LOAD RETRIEVAL SYSTEMS
# =========================================================

print("\nLoading Dense vector store...")

vector_store = load_vector_store()

print("Dense vector store loaded.")


print("\nBuilding BM25 index...")

bm25, bm25_chunks = build_bm25_index()

print("BM25 index ready.")


print("\nLoading Cross-Encoder...")

reranker = load_reranker()

print("Cross-Encoder loaded.")


# =========================================================
# ANALYZE RESULT
# =========================================================

def analyze_result(
    question_id,
    question,
    relevant_doc_ids,
    method,
    top20_results,
    top5_results
):

    top20_documents = get_unique_doc_ranking(
        top20_results
    )

    top5_documents = get_unique_doc_ranking(
        top5_results
    )

    failure = classify_failure(
        top20_documents,
        top5_documents,
        relevant_doc_ids
    )

    ranks = get_relevant_ranks(
        top20_documents,
        relevant_doc_ids
    )

    return {
        "question_id": question_id,
        "question": question,
        "method": method,
        "relevant_doc_ids": relevant_doc_ids,

        "top20_doc_ids": [
            doc.metadata.get("doc_id")
            for doc in top20_documents
        ],

        "top5_doc_ids": [
            doc.metadata.get("doc_id")
            for doc in top5_documents
        ],

        "relevant_ranks": ranks,

        "failure_type": failure["failure_type"],

        "found_in_top20": failure["found_in_top20"],

        "found_in_top5": failure["found_in_top5"],

        "missing_from_top20": failure["missing_from_top20"],

        "pushed_out_of_top5": failure["pushed_out_of_top5"]
    }


# =========================================================
# RESULTS
# =========================================================

results = []


# =========================================================
# MAIN LOOP
# =========================================================

for item in evaluation_dataset:

    question_id = item["id"]

    question = item["question"]

    relevant_doc_ids = item["relevant_doc_ids"]

    print(
        f"\nAnalyzing question "
        f"{question_id}/{len(evaluation_dataset)}..."
    )


    # =====================================================
    # 1. DENSE
    # =====================================================

    dense_results = retrieve_documents(
        vector_store,
        question,
        k=20
    )

    dense_top20 = get_unique_doc_ranking(
        dense_results
    )

    dense_top5 = dense_top20[:5]

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "Dense",
            dense_top20,
            dense_top5
        )
    )


    # =====================================================
    # 2. BM25
    # =====================================================

    bm25_results = retrieve_bm25(
        bm25,
        bm25_chunks,
        question,
        k=20
    )

    bm25_top20 = get_unique_doc_ranking(
        bm25_results
    )

    bm25_top5 = bm25_top20[:5]

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "BM25",
            bm25_top20,
            bm25_top5
        )
    )


    # =====================================================
    # 3. HYBRID RRF
    # =====================================================

    # =====================================================
# 3. HYBRID RRF
# =====================================================

    hybrid_dense = retrieve_documents(
        vector_store,
        question,
        k=20
    )

    hybrid_bm25 = retrieve_bm25(
        bm25,
        bm25_chunks,
        question,
        k=20
    )

    # Get the complete RRF ranking.
    # This is the actual ranking produced by Hybrid Retrieval.

    hybrid_rrf_results = reciprocal_rank_fusion(
        hybrid_dense,
        hybrid_bm25
    )

    hybrid_top20 = get_unique_doc_ranking(
        hybrid_rrf_results
    )

    hybrid_top5 = hybrid_top20[:5]

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "Hybrid RRF",
            hybrid_top20,
            hybrid_top5
        )
    )


    # =====================================================
    # 4. QUERY EXPANSION + DENSE
    # =====================================================

    expanded_query = query_expansion_map[question_id]

    expanded_results = retrieve_documents(
        vector_store,
        expanded_query,
        k=20
    )

    expanded_top20 = get_unique_doc_ranking(
        expanded_results
    )

    expanded_top5 = expanded_top20[:5]

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "Query Expansion + Dense",
            expanded_top20,
            expanded_top5
        )
    )


    # =====================================================
    # 5. HYDE + DENSE
    # =====================================================

    hypothetical_document = hyde_map[question_id]

    hyde_results = retrieve_documents(
        vector_store,
        hypothetical_document,
        k=20
    )

    hyde_top20 = get_unique_doc_ranking(
        hyde_results
    )

    hyde_top5 = hyde_top20[:5]

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "HyDE + Dense",
            hyde_top20,
            hyde_top5
        )
    )


    # =====================================================
    # 6. DENSE + CROSS-ENCODER
    # =====================================================

    reranker_candidates = retrieve_documents(
        vector_store,
        question,
        k=20
    )

    reranked_results = rerank_documents(
        reranker,
        question,
        [
            result[0]
            for result in reranker_candidates
        ],
        top_k=5
    )

    reranked_top5 = get_unique_doc_ranking(
        reranked_results
    )

    reranked_top20 = get_unique_doc_ranking(
        reranker_candidates
    )

    results.append(
        analyze_result(
            question_id,
            question,
            relevant_doc_ids,
            "Dense + Cross-Encoder",
            reranked_top20,
            reranked_top5
        )
    )


# =========================================================
# SAVE
# =========================================================

with open(OUTPUT_PATH, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 60)

print("FAILURE ANALYSIS COMPLETE")

print("=" * 60)

print(
    f"\nSaved results to:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nTotal records: {len(results)}"
)

print(
    f"Expected records: "
    f"{len(evaluation_dataset) * 6}"
)