import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.hyde import generate_hypothetical_document


DATASET_PATH = PROJECT_ROOT / "experiments" / "evaluation_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "hyde_documents.json"


# Load evaluation dataset
with open(DATASET_PATH, "r") as f:
    evaluation_dataset = json.load(f)


# Load existing HyDE generations if they exist
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, "r") as f:
        hyde_documents = json.load(f)
else:
    hyde_documents = []


completed_ids = {
    item["id"]
    for item in hyde_documents
}


for item in evaluation_dataset:

    question_id = item["id"]

    if question_id in completed_ids:
        print(f"Question {question_id} already exists. Skipping.")
        continue

    original_query = item["question"]

    print(f"\nGenerating HyDE document for question {question_id}/30...")

    try:

        hypothetical_document = generate_hypothetical_document(
            original_query
        )

        result = {
            "id": question_id,
            "original_query": original_query,
            "hypothetical_document": hypothetical_document
        }

        hyde_documents.append(result)

        # Save immediately
        with open(OUTPUT_PATH, "w") as f:
            json.dump(
                hyde_documents,
                f,
                indent=4
            )

        print("\nHypothetical document:")
        print(hypothetical_document)

        print("\nSaved successfully.")

        print("Waiting 15 seconds before next request...")
        time.sleep(15)

    except Exception as e:

        print(
            f"\nError processing question {question_id}:"
        )

        print(e)

        print(
            "\nStopping. Already completed questions have been saved."
        )

        break


print("\nHyDE generation process finished.")
print(f"Saved to: {OUTPUT_PATH}")