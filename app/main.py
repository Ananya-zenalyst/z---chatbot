from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.chat_agent import get_chat_agent
from app.schemas.models import ChatRequest, ChatResponse, UploadResponse
import logging
import traceback

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

    The files are processed, chunked, and stored in a vector database.
    """
    global is_vector_store_ready
    try:
        logger.info(f"Received {len(files)} files for processing.")
        # Process all files concurrently
        contents = [await file.read() for file in files]
        filenames = [file.filename for file in files]

        processor = DocumentProcessor()
        documents = await processor.process_documents(contents, filenames)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract any content from the provided files.")

        # Load existing store or create a new one with the processed documents
        VectorStoreService.load_or_create_vector_store(documents)
        is_vector_store_ready = True

        logger.info(f"Successfully processed and vectorized {len(documents)} document chunks.")
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded and processed {len(files)} files.",
                "filenames": filenames
            }
        )
    except Exception as e:
        logger.error(f"Error during file upload and processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/chat/", response_model=ChatResponse, tags=["Chat"])
async def chat_with_agent(request: ChatRequest = Body(...)):
    """
    Endpoint to send a message to the chat agent.

    Requires a `session_id` to maintain conversation history and a `query`.
    """
    if not is_vector_store_ready:
        raise HTTPException(status_code=400, detail="No documents have been uploaded or processed yet. Please upload files first.")

    try:
        logger.info(f"Received chat request for session_id: '{request.session_id}'")
        chat_agent = get_chat_agent()
        response = await chat_agent.get_response(request.query, request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            response=response.get("output", "Sorry, I could not generate a response.")
        )
    except Exception as e:
        logger.error(f"Error during chat processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the chat session: {e}")
