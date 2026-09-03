#before we move forward it is very important to understand what we are embedding.
#all-MiniLM-L6-v2, converts each chunk into a 384 dimension vector

# Chunk 1 → [384 numbers]
# Chunk 2 → [384 numbers]
# Chunk 3 → [384 numbers]
# ...
# Chunk 120 → [384 numbers]

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )
    return embedding_model

if __name__ == "__main__":
    from ingest import load_pubmed_documents
    from chunking import chunk_documents
    #Load karo deocuments ko phele
    docs=load_pubmed_documents("pubmed_articles")

    #Create Chunks
    chunks= chunk_documents(
        docs,
        chunk_size=512,
        chunk_overlap=50
    )

    #load_embedding model

    embedding_model=get_embedding_model()

    #text ectract karna hai har chunk se 
    chunk_text=[
        chunk.page_content
        for chunk in chunks
    ]

    #ab har chunk ke liye embeddings generate karo

    vector=embedding_model.embed_documents(chunk_text)

    print(f"Total chunks: {len(chunks)}")
    print(f"Total vectors: {len(vector)}")
    print(f"Embedding dimension: {len(vector[0])}")
    # print(vector[0])