import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.query_expansion import expand_query


DATASET_PATH = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "query_expansions.json"


# Load benchmark
with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)


# Load existing expansions if the file already exists
if OUTPUT_PATH.exists():

    with open(OUTPUT_PATH, "r") as f:
        expanded_queries = json.load(f)

else:
    expanded_queries = []


# IDs that have already been processed
completed_ids = {
    item["id"]
    for item in expanded_queries
}


for item in evaluation_dataset:

    question_id = item["id"]

    # Skip questions that were already successfully processed
    if question_id in completed_ids:
        print(f"Question {question_id} already exists. Skipping.")
        continue

    original_query = item["question"]

    print(f"\nExpanding question {question_id}/30...")

    try:

        expanded_query = expand_query(original_query)

        result = {
            "id": question_id,
            "original_query": original_query,
            "expanded_query": expanded_query
        }

        expanded_queries.append(result)

        # Save immediately
        with open(OUTPUT_PATH, "w") as f:
            json.dump(expanded_queries, f, indent=4)

        print("Expanded query:")
        print(expanded_query)

        print("Saved successfully.")

        # Stay below the free-tier RPM limit
        print("Waiting 15 seconds before next request...")
        time.sleep(15)

    except Exception as e:

        print(f"\nError processing question {question_id}:")
        print(e)

        print("\nStopping. Already completed questions have been saved.")
        break


print("\nExpansion process finished.")
print(f"Saved to: {OUTPUT_PATH}")