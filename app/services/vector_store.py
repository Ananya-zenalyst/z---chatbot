from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Optional
from langchain.vectorstores.base import VectorStoreRetriever
from app.core.config import settings
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Manages the creation, loading, and querying of the vector store.

    This service handles the embedding of document chunks and their storage
    in a FAISS vector database, which is persisted to the local disk.
    """
    _vector_store: Optional[FAISS] = None
    _embeddings_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    _persist_directory: str = "vector_storage"

    @classmethod
    def get_instance(cls):
        """
        Provides a singleton instance of the VectorStoreService.
        This ensures that the vector store is initialized only once.
        """
        if cls._vector_store is None:
            cls.load_or_create_vector_store()
        return cls

    @classmethod
    def load_or_create_vector_store(cls, documents: Optional[List[Document]] = None):
        """
        Loads the vector store from disk if it exists, otherwise creates a new one.

        Args:
            documents (Optional[List[Document]]): A list of documents to add to a new
                                                 vector store. Required if the store
                                                 doesn't exist on disk.
        """
        if os.path.exists(cls._persist_directory) and os.listdir(cls._persist_directory):
            logger.info(f"Loading existing vector store from '{cls._persist_directory}'.")
            cls._vector_store = FAISS.load_local(
                cls._persist_directory,
                cls._embeddings_model,
                allow_dangerous_deserialization=True  # Required for FAISS with LangChain
            )
            logger.info("Vector store loaded successfully.")
            if documents:
                logger.info(f"Adding {len(documents)} new documents to the existing store.")
                # Add documents in batches to avoid token limit errors
                cls._add_documents_in_batches(documents)
                cls._vector_store.save_local(cls._persist_directory)
                logger.info("New documents added and store updated.")
        elif documents:
            logger.info(f"Creating a new vector store with {len(documents)} documents.")
            os.makedirs(cls._persist_directory, exist_ok=True)
            # Process first batch to create the store
            batch_size = 50
            first_batch = documents[:batch_size]
            cls._vector_store = FAISS.from_documents(first_batch, cls._embeddings_model)

            # Add remaining documents in batches
            if len(documents) > batch_size:
                remaining_docs = documents[batch_size:]
                cls._add_documents_in_batches(remaining_docs)

            cls._vector_store.save_local(cls._persist_directory)
            logger.info(f"New vector store created and saved to '{cls._persist_directory}'.")
        else:
            logger.warning("Vector store does not exist and no documents were provided to create a new one.")
            cls._vector_store = None

    @classmethod
    def _add_documents_in_batches(cls, documents: List[Document], batch_size: int = 50):
        """
        Add documents to the vector store in batches to avoid token limit errors.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process at once
        """
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            end_idx = min(i + batch_size, total_docs)
            logger.info(f"Processing batch {i//batch_size + 1}: documents {i+1} to {end_idx}")
            try:
                cls._vector_store.add_documents(batch)
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                # Try with smaller batch size if it fails
                if batch_size > 10:
                    logger.info("Retrying with smaller batch size...")
                    for doc in batch:
                        try:
                            cls._vector_store.add_documents([doc])
                        except Exception as e2:
                            logger.error(f"Failed to add document: {e2}")
                            continue


    @classmethod
    def get_retriever(cls, search_type="mmr", k=10) -> Optional[VectorStoreRetriever]:
        """
        Enhanced retriever with better coverage and relevance scoring.

        Args:
            search_type (str): The type of search to perform (e.g., "similarity", "mmr").
            k (int): Increased to 10 for better information coverage.

        Returns:
            Optional[VectorStoreRetriever]: A retriever instance if the store exists,
                                            otherwise None.
        """
        instance = cls.get_instance()
        if instance._vector_store:
            # Use MMR (Maximum Marginal Relevance) for better diversity
            if search_type == "mmr":
                return instance._vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,  # Return more documents for better coverage
                        "fetch_k": 30,  # Fetch even more candidates for MMR
                        "lambda_mult": 0.3  # Prioritize diversity more (lower = more diverse)
                    }
                )
            else:
                return instance._vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": k}
                )
        logger.error("Could not create retriever because vector store is not initialized.")
        return None

    @classmethod
    def search_with_score(cls, query: str, k: int = 10):
        """
        Search with relevance scores for better result filtering.
        """
        instance = cls.get_instance()
        if instance._vector_store:
            return instance._vector_store.similarity_search_with_score(query, k=k)
        return []

    @classmethod
    def hybrid_search(cls, query: str, k: int = 15):
        """
        Perform hybrid search combining similarity and keyword matching.
        """
        instance = cls.get_instance()
        if not instance._vector_store:
            return []

        # Get similarity results
        similarity_results = instance._vector_store.similarity_search(query, k=k)

        # Get MMR results for diversity
        mmr_results = instance._vector_store.max_marginal_relevance_search(
            query, k=k//2, fetch_k=20
        )

        # Combine and deduplicate
        seen_content = set()
        combined_results = []

        for doc in similarity_results + mmr_results:
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_results.append(doc)

        return combined_results[:k]
