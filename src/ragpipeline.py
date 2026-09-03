from retriever import load_vector_store, retrieve_documents
from generator import generate_answer

def run_rag(query,k=5):

    #Phele we build a vector store
    vector_store=load_vector_store()

    #Retrieve relevant chunks
    results= retrieve_documents(
        vector_store,
        query,
        k=k
    )

    #extract documents

    retrieved_documents=[
        doc for doc, score in results
    ]

    #generate answer using retrieved context

    answer=generate_answer(
        query,
        retrieved_documents
    )

    return answer, results


if __name__ == "__main__":

    query = input("\nEnter your question: ")

    answer, results = run_rag(
        query,
        k=5
    )

    print("\n==============================")
    print("RETRIEVED DOCUMENTS for debugging purposes")
    print("==============================\n")

    for i, (doc, score) in enumerate(results):

        print(f"--- Result {i + 1} ---")
        print(f"Score: {score}")
        print(f"Doc ID: {doc.metadata.get('doc_id')}")
        print(f"Source: {doc.metadata.get('file_name')}")
        print(doc.page_content[:500])
        print()

    print("\n==============================")
    print("GENERATED ANSWER")
    print("==============================\n")

    print(answer)