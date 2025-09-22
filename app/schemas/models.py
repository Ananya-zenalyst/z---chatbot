from pydantic import BaseModel, Field
from typing import List
import uuid

class ChatRequest(BaseModel):
    """
    Defines the structure for a chat request from the client.
    """
    query: str = Field(..., description="The user's question for the chatbot.")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="A unique identifier for the chat session. If not provided, a new one will be generated."
    )

class DocumentInfo(BaseModel):
    """
    Provides metadata about a successfully uploaded document.
    """
    filename: str
    content_type: str

class UploadResponse(BaseModel):
    """
    Defines the structure for the response after files are uploaded.
    """
    message: str
    processed_files: List[DocumentInfo]

class ChatResponse(BaseModel):
    """
    Defines the structure for the chat response to the client.
    """
    session_id: str = Field(..., description="The unique identifier for the chat session.")
    response: str = Field(..., description="The chatbot's response to the user's query.")
