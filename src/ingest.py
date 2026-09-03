import os
import json
from langchain_core.documents import Document


def load_pubmed_documents(data_folder: str):
    """
    Loads all PubMed JSON files and converts them into LangChain Documents.
    """

    documents = [] #intilaly this is an empty list.

    for filename in os.listdir(data_folder):

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(data_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            article = json.load(f)

        content = f"""
        Title: {article.get("title", "")}

        Abstract:
        {article.get("abstract", "")}

        Keywords:
        {", ".join(article.get("keywords", []))}
        """

        doc = Document(
            page_content=content.strip(),
            metadata={
                "doc_id": article.get("doc_id", filename),
                "source": "PubMed",
                "file_name": filename
            }
        )

        documents.append(doc)

    # print(f"Loaded {len(documents)} documents.")

    return documents


if __name__ == "__main__":

    docs = load_pubmed_documents("pubmed_articles")
    print(f"Loaded {len(docs)} documents")

    if docs:
        print("\nSample Document:\n")
        print(docs[0].page_content[:500])
    # print("\nSample Document:\n")
    # print(docs[0].page_content[:500])