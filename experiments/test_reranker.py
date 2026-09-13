import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.retriever import load_vector_store, retrieve_documents
from src.reranker import load_reranker, rerank_documents


query = "What causes beta-cell dysfunction in type 2 diabetes?"

print("Loading vector store...")
vector_store = load_vector_store()

print("Retrieving Top-20 candidates...")
results = retrieve_documents(
    vector_store,
    query,
    k=20
)

# retrieve_documents returns:
# [(Document, similarity_score), ...]

documents = [doc for doc, score in results]

print("\nDense Retrieval Ranking:")
for i, doc in enumerate(documents[:5], start=1):
    print(
        f"{i}. {doc.metadata.get('doc_id')} "
        f"| {doc.page_content[:100]}..."
    )


print("\nLoading Cross-Encoder...")
reranker = load_reranker()

print("Reranking candidates...")

reranked = rerank_documents(
    reranker,
    query,
    documents,
    top_k=5
)

print("\nAfter Cross-Encoder Reranking:")

for i, (doc, score) in enumerate(reranked, start=1):
    print(
        f"{i}. {doc.metadata.get('doc_id')} "
        f"| Score: {score:.4f}"
    )