import os
import uuid
import shutil
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from langchain.docstore.document import Document
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages document sessions with automatic cleanup and isolation.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.base_storage_dir = "session_storage"
        os.makedirs(self.base_storage_dir, exist_ok=True)

    def create_session(self) -> str:
        """Create a new session with unique ID."""
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(self.base_storage_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "storage_dir": session_dir,
            "documents": [],
            "vector_store": None
        }

        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session_storage_dir(self, session_id: str) -> str:
        """Get the storage directory for a session."""
        return os.path.join(self.base_storage_dir, session_id)

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions

    def update_last_accessed(self, session_id: str):
        """Update the last accessed time for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["last_accessed"] = datetime.now()

    def add_documents_to_session(self, session_id: str, documents: List[Document]):
        """Add documents to a specific session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} does not exist")

        session_data = self.sessions[session_id]
        session_data["documents"].extend(documents)

        # Create session-specific vector store
        session_vector_dir = os.path.join(session_data["storage_dir"], "vector_store")

        # Create a temporary VectorStoreService instance for this session
        session_vector_store = SessionVectorStore(session_vector_dir)
        session_vector_store.load_or_create_vector_store(documents)

        session_data["vector_store"] = session_vector_store
        self.update_last_accessed(session_id)

        logger.info(f"Added {len(documents)} documents to session {session_id}")

    def get_session_vector_store(self, session_id: str) -> Optional['SessionVectorStore']:
        """Get the vector store for a specific session."""
        if session_id in self.sessions:
            self.update_last_accessed(session_id)
            return self.sessions[session_id].get("vector_store")
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        if session_id not in self.sessions:
            return False

        session_data = self.sessions[session_id]
        storage_dir = session_data["storage_dir"]

        # Remove storage directory
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)

        # Remove from memory
        del self.sessions[session_id]

        logger.info(f"Deleted session {session_id}")
        return True

    def cleanup_inactive_sessions(self, inactive_minutes: int = 5):
        """Clean up sessions that have been inactive for specified minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=inactive_minutes)
        sessions_to_delete = []

        for session_id, session_data in self.sessions.items():
            if session_data["last_accessed"] < cutoff_time:
                sessions_to_delete.append(session_id)

        for session_id in sessions_to_delete:
            self.delete_session(session_id)
            logger.info(f"Auto-deleted inactive session: {session_id}")

        return len(sessions_to_delete)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session."""
        if session_id in self.sessions:
            session_data = self.sessions[session_id].copy()
            session_data["document_count"] = len(session_data["documents"])
            session_data["has_vector_store"] = session_data["vector_store"] is not None
            # Remove the actual documents and vector store from the returned info
            session_data.pop("documents", None)
            session_data.pop("vector_store", None)
            return session_data
        return None


class SessionVectorStore:
    """
    Session-specific vector store that isolates documents per session.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.vector_store = None
        from langchain_openai import OpenAIEmbeddings
        from app.core.config import settings
        self.embeddings_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    def load_or_create_vector_store(self, documents: Optional[List[Document]] = None):
        """Load or create vector store for this session."""
        from langchain_community.vectorstores import FAISS

        if os.path.exists(self.storage_dir) and os.listdir(self.storage_dir):
            logger.info(f"Loading existing session vector store from '{self.storage_dir}'.")
            self.vector_store = FAISS.load_local(
                self.storage_dir,
                self.embeddings_model,
                allow_dangerous_deserialization=True
            )
            if documents:
                logger.info(f"Adding {len(documents)} new documents to session store.")
                self._add_documents_in_batches(documents)
                self.vector_store.save_local(self.storage_dir)
        elif documents:
            logger.info(f"Creating new session vector store with {len(documents)} documents.")
            os.makedirs(self.storage_dir, exist_ok=True)

            # Process documents in batches
            batch_size = 50
            first_batch = documents[:batch_size]
            self.vector_store = FAISS.from_documents(first_batch, self.embeddings_model)

            if len(documents) > batch_size:
                remaining_docs = documents[batch_size:]
                self._add_documents_in_batches(remaining_docs)

            self.vector_store.save_local(self.storage_dir)
            logger.info(f"Session vector store saved to '{self.storage_dir}'.")

    def _add_documents_in_batches(self, documents: List[Document], batch_size: int = 50):
        """Add documents in batches to avoid token limits."""
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            try:
                self.vector_store.add_documents(batch)
            except Exception as e:
                logger.error(f"Error adding batch to session store: {e}")
                # Try individual documents if batch fails
                for doc in batch:
                    try:
                        self.vector_store.add_documents([doc])
                    except Exception as e2:
                        logger.error(f"Failed to add document to session store: {e2}")

    def get_retriever(self, search_type="mmr", k=10):
        """Get retriever for this session's vector store."""
        if self.vector_store:
            if search_type == "mmr":
                return self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": 30,
                        "lambda_mult": 0.3
                    }
                )
            else:
                return self.vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": k}
                )
        return None

    def search_with_score(self, query: str, k: int = 10):
        """Search with relevance scores."""
        if self.vector_store:
            return self.vector_store.similarity_search_with_score(query, k=k)
        return []


# Global session manager instance
session_manager = SessionManager()

# Background task for cleanup
async def cleanup_task():
    """Background task to clean up inactive sessions."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        try:
            deleted_count = session_manager.cleanup_inactive_sessions(5)
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} inactive sessions")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")