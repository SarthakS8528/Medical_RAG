import os
import json
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.chunking import chunk_documents


ARTICLES_DIR = "pubmed_articles"


def load_pubmed_documents(directory):
    """
    Load PubMed JSON files and convert them into
    LangChain Document objects.
    """

    documents = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        content = (
            f"Title: {article['title']}\n"
            f"Abstract: {article['abstract']}\n"
            f"Keywords: {', '.join(article.get('keywords', []))}"
        )

        metadata = {
            "doc_id": article["doc_id"],
            "pmid": article.get("pmid"),
            "title": article["title"],
            "source": article.get("source", "PubMed")
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents


def build_bm25_index():

    print("Loading PubMed documents...")

    documents = load_pubmed_documents(ARTICLES_DIR)

    print(f"Loaded {len(documents)} documents.")

    print("Chunking documents...")

    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Store chunk text
    corpus = [chunk.page_content for chunk in chunks]

    # BM25 requires tokenized documents
    tokenized_corpus = [
        text.lower().split()
        for text in corpus
    ]

    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index ready.")

    return bm25, chunks


def retrieve_bm25(bm25, chunks, query, k=20):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    # Rank indices by descending BM25 score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:k]:

        results.append(
            (
                chunks[index],
                float(scores[index])
            )
        )

    return results


if __name__ == "__main__":

    bm25, chunks = build_bm25_index()

    query = "What are the main biological problems that cause blood sugar regulation to break down in type 2 diabetes?"

    results = retrieve_bm25(
        bm25,
        chunks,
        query,
        k=5
    )

    print("\nTop BM25 results:\n")

    for rank, (doc, score) in enumerate(results, start=1):

        print(f"Rank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Doc ID: {doc.metadata.get('doc_id')}")
        print(f"Text: {doc.page_content[:300]}")
        print("-" * 70)