from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from config import Config, logger

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the configured embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cpu'}
    )

def build_vector_store(transcript: str) -> Chroma:
    """
    Builds a Chroma vector store from the given transcript.
    
    Args:
        transcript (str): The transcript to store.
        
    Returns:
        Chroma: The instantiated vector store.
    """
    logger.info("Building vector store from transcript.")
    try:
        # Increased chunk size and overlap for better RAG context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(transcript)

        if not chunks:
            chunks = ["(No transcript content available to index.)"]

        docs = [
            Document(page_content=chunk, metadata={'chunk_index': i})
            for i, chunk in enumerate(chunks)
        ]

        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="meeting_transcript",
            persist_directory=Config.CHROMA_DB_DIR
        )

        logger.info(f"Vector store built successfully with {len(docs)} chunks.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise

def load_vector_store() -> Chroma:
    """Loads an existing vector store from disk."""
    logger.info("Loading vector store from disk.")
    embeddings = get_embeddings()
    return Chroma(
        collection_name="meeting_transcript",
        embedding_function=embeddings,
        persist_directory=Config.CHROMA_DB_DIR
    )

def get_retriever(vector_store: Chroma, k: int = 10) -> VectorStoreRetriever:
    """Returns a retriever for the given vector store."""
    return vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={"k": k}
    )
