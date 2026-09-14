import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "failure_analysis_results.json"
)


with open(INPUT_PATH, "r") as f:
    results = json.load(f)


# =========================================================
# 1. Failure counts by method
# =========================================================

methods = sorted(
    set(result["method"] for result in results)
)


print("\n" + "=" * 70)
print("FAILURE SUMMARY BY METHOD")
print("=" * 70)


for method in methods:

    method_results = [
        r for r in results
        if r["method"] == method
    ]

    counts = Counter(
        r["failure_type"]
        for r in method_results
    )

    print(f"\n{method}")
    print("-" * 50)

    for failure_type in [
        "Success",
        "Ranking Failure",
        "Retrieval Failure",
        "Multi-document Recall Failure"
    ]:

        print(
            f"{failure_type:<30}: "
            f"{counts.get(failure_type, 0)}"
        )


# =========================================================
# 2. Questions that failed consistently
# =========================================================

print("\n" + "=" * 70)
print("QUESTIONS THAT FAILED ACROSS METHODS")
print("=" * 70)


question_results = {}

for result in results:

    qid = result["question_id"]

    question_results.setdefault(
        qid,
        {
            "question": result["question"],
            "methods": []
        }
    )

    question_results[qid]["methods"].append(result)


for qid, data in sorted(question_results.items()):

    failures = [
        r for r in data["methods"]
        if r["failure_type"] != "Success"
    ]

    if len(failures) >= 4:

        print(f"\nQ{qid}: {data['question']}")

        for r in failures:

            print(
                f"  {r['method']:<30} "
                f"{r['failure_type']}"
            )


# =========================================================
# 3. Questions successfully solved by one method
#    but failed by another
# =========================================================

print("\n" + "=" * 70)
print("METHOD-SPECIFIC SUCCESS / FAILURE")
print("=" * 70)


for qid, data in sorted(question_results.items()):

    successes = [
        r["method"]
        for r in data["methods"]
        if r["failure_type"] == "Success"
    ]

    failures = [
        r["method"]
        for r in data["methods"]
        if r["failure_type"] != "Success"
    ]

    if successes and failures:

        print(f"\nQ{qid}: {data['question']}")

        print(
            "  Successful: "
            + ", ".join(successes)
        )

        print(
            "  Failed: "
            + ", ".join(failures)
        )


# =========================================================
# 4. Most common missing documents
# =========================================================

print("\n" + "=" * 70)
print("MOST FREQUENTLY MISSED RELEVANT DOCUMENTS")
print("=" * 70)


missing_docs = Counter()


for result in results:

    for doc_id in result["missing_from_top20"]:

        missing_docs[doc_id] += 1


for doc_id, count in missing_docs.most_common(15):

    print(
        f"{doc_id:<15}: "
        f"missed {count} times"
    )


# =========================================================
# 5. Ranking problems
# =========================================================

print("\n" + "=" * 70)
print("RANKING FAILURES")
print("=" * 70)


ranking_failures = []


for result in results:

    if result["failure_type"] == "Ranking Failure":

        ranking_failures.append(result)


for result in ranking_failures:

    print(
        f"\nQ{result['question_id']} "
        f"| {result['method']}"
    )

    print(
        f"Question: {result['question']}"
    )

    print(
        f"Relevant docs: "
        f"{result['relevant_doc_ids']}"
    )

    print(
        f"Relevant ranks: "
        f"{result['relevant_ranks']}"
    )

    print(
        f"Top-5: "
        f"{result['top5_doc_ids']}"
    )


# =========================================================
# 6. Multi-document recall failures
# =========================================================

print("\n" + "=" * 70)
print("MULTI-DOCUMENT RECALL FAILURES")
print("=" * 70)


multi_doc_failures = []


for result in results:

    if result["failure_type"] == "Multi-document Recall Failure":

        multi_doc_failures.append(result)


for result in multi_doc_failures:

    print(
        f"\nQ{result['question_id']} "
        f"| {result['method']}"
    )

    print(
        f"Question: {result['question']}"
    )

    print(
        f"Relevant docs: "
        f"{result['relevant_doc_ids']}"
    )

    print(
        f"Found in Top-20: "
        f"{result['found_in_top20']}"
    )

    print(
        f"Found in Top-5: "
        f"{result['found_in_top5']}"
    )

    print(
        f"Missing from Top-20: "
        f"{result['missing_from_top20']}"
    )

    print(
        f"Pushed outside Top-5: "
        f"{result['pushed_out_of_top5']}"
    )