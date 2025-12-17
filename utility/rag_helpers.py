from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def make_text_splitter(chunk_size: int, chunk_overlap: int) -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = make_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)


def get_local_embeddings():
    # Local embeddings (FAISS-safe)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_faiss_from_texts(
    text_items,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
):
    """
    text_items: iterable of (source_name, text_str)
    Returns: FAISS vectorstore
    """
    splitter = make_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    metadatas = []
    texts = []
    for source, text in text_items:
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks, start=1):
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if not texts:
        raise ValueError("No text chunks found to index.")

    embeddings = get_local_embeddings()
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
