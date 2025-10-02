from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.chat_agent import get_chat_agent
from app.services.session_manager import session_manager, cleanup_task
from app.schemas.models import ChatRequest, ChatResponse, UploadResponse, DeleteSessionRequest, DeleteSessionResponse
import logging
import traceback
import asyncio
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A flag to check if the vector store is ready for querying.
is_vector_store_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.
    """
    # Startup
    global is_vector_store_ready
    try:
        # Attempt to get an instance, which will load from disk if available.
        VectorStoreService.get_instance()
        if VectorStoreService._vector_store is not None:
            is_vector_store_ready = True
            logger.info("Application startup: Vector store loaded and ready.")
        else:
            logger.info("Application startup: No existing vector store found. Waiting for uploads.")
    except Exception as e:
        logger.error(f"Error loading vector store at startup: {e}")

    yield
    # Shutdown
    logger.info("Application shutting down.")

# Initialize FastAPI app
app = FastAPI(
    title="Financial Chatbot API",
    description="An API for interacting with a financial chatbot that can process PDFs and search the web.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}


@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload one or more PDF files.

    Creates a new session and processes files for that specific session.
    """
    try:
        logger.info(f"Received {len(files)} files for processing.")

        # Create a new session
        session_id = session_manager.create_session()

        # Process all files concurrently
        contents = [await file.read() for file in files]
        filenames = [file.filename for file in files]

        processor = DocumentProcessor()
        documents = await processor.process_documents(contents, filenames)

        if not documents:
            # Clean up the session if no documents could be processed
            session_manager.delete_session(session_id)
            raise HTTPException(status_code=400, detail="Could not extract any content from the provided files.")

        # Add documents to the session
        session_manager.add_documents_to_session(session_id, documents)

        logger.info(f"Successfully processed and vectorized {len(documents)} document chunks for session {session_id}.")
        return UploadResponse(
            message=f"Successfully uploaded and processed {len(files)} files.",
            session_id=session_id,
            filenames=filenames
        )
    except Exception as e:
        logger.error(f"Error during file upload and processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/chat/", response_model=ChatResponse, tags=["Chat"])
async def chat_with_agent(request: ChatRequest = Body(...)):
    """
    Endpoint to send a message to the chat agent.

    Requires a `session_id` to maintain conversation history and a `query`.
    Uses session-specific documents for responses.
    """
    try:
        # Check if session exists
        if not session_manager.session_exists(request.session_id):
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found. Please upload documents first."
            )

        # Get session-specific vector store
        session_vector_store = session_manager.get_session_vector_store(request.session_id)
        if not session_vector_store:
            raise HTTPException(
                status_code=400,
                detail=f"No documents found for session {request.session_id}. Please upload documents first."
            )

        logger.info(f"Received chat request for session_id: '{request.session_id}'")

        # Get chat agent with session-specific retriever
        from app.services.chat_agent import get_session_chat_agent
        chat_agent = get_session_chat_agent(session_vector_store)
        response = await chat_agent.get_response(request.query, request.session_id)

        return ChatResponse(
            session_id=request.session_id,
            response=response.get("output", "Sorry, I could not generate a response.")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chat processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the chat session: {e}")


@app.post("/delete/", response_model=DeleteSessionResponse, tags=["Sessions"])
async def delete_session(request: DeleteSessionRequest = Body(...)):
    """
    Endpoint to delete a session and all its associated data.
    """
    try:
        if not session_manager.session_exists(request.session_id):
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found."
            )

        success = session_manager.delete_session(request.session_id)
        if success:
            return DeleteSessionResponse(
                message=f"Session {request.session_id} successfully deleted.",
                session_id=request.session_id
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete session {request.session_id}."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during session deletion: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during session deletion: {e}")


@app.get("/sessions/{session_id}/info", tags=["Sessions"])
async def get_session_info(session_id: str):
    """
    Endpoint to get information about a specific session.
    """
    try:
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found."
            )
        return session_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred while getting session info: {e}")


@app.delete("/sessions/cleanup", tags=["Sessions"])
async def cleanup_inactive_sessions(inactive_minutes: int = 5):
    """
    Endpoint to manually trigger cleanup of inactive sessions.
    """
    try:
        deleted_count = session_manager.cleanup_inactive_sessions(inactive_minutes)
        return {
            "message": f"Cleaned up {deleted_count} inactive sessions.",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error during manual cleanup: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during cleanup: {e}")
