def get_unique_doc_ranking(results):
    """
    Convert chunk-level retrieval results into a document-level ranking.

    Results may be either:
        Document
    or:
        (Document, score)

    The first occurrence of a doc_id is kept because it represents
    the highest-ranked chunk belonging to that document.
    """

    seen = set()
    unique_documents = []

    for result in results:

        if isinstance(result, tuple):
            doc = result[0]
        else:
            doc = result

        doc_id = doc.metadata.get("doc_id")

        if doc_id not in seen:
            seen.add(doc_id)
            unique_documents.append(doc)

    return unique_documents


def get_relevant_ranks(retrieved_documents, relevant_doc_ids):
    """
    Return the rank of every relevant document that was retrieved.
    """

    relevant_doc_ids = set(relevant_doc_ids)

    ranks = {}

    for rank, doc in enumerate(retrieved_documents, start=1):

        doc_id = doc.metadata.get("doc_id")

        if doc_id in relevant_doc_ids:
            ranks[doc_id] = rank

    return ranks


def classify_failure(
    retrieved_top20,
    retrieved_top5,
    relevant_doc_ids
):
    """
    Classify retrieval behavior into:

    - Success
    - Ranking Failure
    - Retrieval Failure
    - Multi-document Recall Failure

    Top-20 represents the candidate pool.
    Top-5 represents the final retrieved set.
    """

    relevant_doc_ids = set(relevant_doc_ids)

    top20_ids = {
        doc.metadata.get("doc_id")
        for doc in retrieved_top20
    }

    top5_ids = {
        doc.metadata.get("doc_id")
        for doc in retrieved_top5
    }

    found_in_top20 = relevant_doc_ids & top20_ids
    found_in_top5 = relevant_doc_ids & top5_ids

    missing_from_top20 = relevant_doc_ids - top20_ids
    pushed_out_of_top5 = found_in_top20 - found_in_top5

    if len(relevant_doc_ids) > 1 and len(found_in_top5) < len(relevant_doc_ids):
        failure_type = "Multi-document Recall Failure"

    elif missing_from_top20:
        failure_type = "Retrieval Failure"

    elif pushed_out_of_top5:
        failure_type = "Ranking Failure"

    else:
        failure_type = "Success"

    return {
        "failure_type": failure_type,
        "found_in_top20": sorted(found_in_top20),
        "found_in_top5": sorted(found_in_top5),
        "missing_from_top20": sorted(missing_from_top20),
        "pushed_out_of_top5": sorted(pushed_out_of_top5)
    }