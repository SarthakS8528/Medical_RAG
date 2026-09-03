from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=512,chunk_overlap=50):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks=splitter.split_documents(documents)
    return chunks

if __name__=="__main__":
    from ingest import load_pubmed_documents
    docs=load_pubmed_documents("pubmed_articles")

    chunks=chunk_documents(
        docs,chunk_size=512,chunk_overlap=50)
    print(f"created {len(chunks)} chunks.")
    # rint("\n Sample Chunk aayyega neeche")p
    print(chunks[0].page_content[:1000])
    
    print("\nMetadata:\n")
    print(chunks[0].metadata)
