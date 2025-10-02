import streamlit as st
import asyncio
import requests
import json
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.chat_agent import get_chat_agent
import logging
import os
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Z Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50 !important;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .chat-message.user {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #2c3e50;
        margin-right: 20%;
        border: 2px solid #e9ecef;
    }
    .chat-message .message {
        width: 100%;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Welcome section */
    .upload-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: #2c3e50;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: #2c3e50;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #2c3e50;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        font-size: 16px;
        caret-color: black !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }

    /* File uploader */
    .uploadedFile {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
    }
    .stInfo {
        background-color: #cce7ff;
        color: #004085;
        border: 1px solid #b3d9ff;
        border-radius: 10px;
    }

    /* Title and headers */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Sidebar text */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg p {
        color: #ffffff !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        color: #ffffff;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Spinner */
    .stSpinner {
        color: #ffffff;
    }

    /* Container backgrounds */
    .block-container {
        background-color: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }

    /* Session info styling */
    .session-info {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False

# API Helper Functions
def upload_files_to_api(files):
    """Upload files to the FastAPI backend and get session ID"""
    files_data = []
    for file in files:
        files_data.append(('files', (file.name, file.getvalue(), 'application/pdf')))

    try:
        response = requests.post(f"{API_BASE_URL}/upload/", files=files_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def send_chat_message(session_id, query):
    """Send chat message to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json={"session_id": session_id, "query": query}
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error("Session not found. Please upload documents again.")
            return None
        else:
            st.error(f"Chat failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def delete_session(session_id):
    """Delete session manually"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/delete/",
            json={"session_id": session_id}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Delete failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def get_session_info(session_id):
    """Get session information"""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None

# Check if vector store exists on startup
@st.cache_resource
def initialize_vector_store():
    """Initialize or load existing vector store"""
    try:
        VectorStoreService.get_instance()
        if VectorStoreService._vector_store is not None:
            return True
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
    return False

# Session Management JavaScript
session_js = """
<script>
class StreamlitSessionManager {
    constructor() {
        this.sessionId = null;
        this.isActive = true;
        this.inactivityTimer = null;
        this.INACTIVITY_TIMEOUT = 5 * 60 * 1000; // 5 minutes
        this.API_BASE_URL = 'http://localhost:8000';
        this.init();
    }

    init() {
        this.setupActivityListeners();
        this.setupVisibilityListener();
        this.setupBeforeUnloadListener();
        console.log('Session Manager initialized');
    }

    setSessionId(sessionId) {
        this.sessionId = sessionId;
        sessionStorage.setItem('streamlit_session_id', sessionId);
        console.log('Session ID set:', sessionId);
    }

    getSessionId() {
        if (!this.sessionId) {
            this.sessionId = sessionStorage.getItem('streamlit_session_id');
        }
        return this.sessionId;
    }

    setupActivityListeners() {
        window.addEventListener('focus', () => this.onActivityResume());
        window.addEventListener('blur', () => this.onActivityPause());
        const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
        activityEvents.forEach(event => {
            document.addEventListener(event, () => this.resetInactivityTimer(), true);
        });
    }

    setupVisibilityListener() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.onActivityPause();
            } else {
                this.onActivityResume();
            }
        });
    }

    setupBeforeUnloadListener() {
        window.addEventListener('beforeunload', () => {
            this.cleanupSession();
        });
    }

    onActivityResume() {
        this.isActive = true;
        this.clearInactivityTimer();
        console.log('Activity resumed');
    }

    onActivityPause() {
        this.isActive = false;
        this.startInactivityTimer();
        console.log('Activity paused');
    }

    startInactivityTimer() {
        this.clearInactivityTimer();
        this.inactivityTimer = setTimeout(() => {
            console.log('Session inactive for 5 minutes, cleaning up');
            this.cleanupSession();
        }, this.INACTIVITY_TIMEOUT);
    }

    clearInactivityTimer() {
        if (this.inactivityTimer) {
            clearTimeout(this.inactivityTimer);
            this.inactivityTimer = null;
        }
    }

    resetInactivityTimer() {
        if (!this.isActive) return;
        this.clearInactivityTimer();
        this.startInactivityTimer();
    }

    async cleanupSession() {
        if (!this.sessionId) return;
        try {
            await fetch(`${this.API_BASE_URL}/delete/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            console.log('Session cleaned up:', this.sessionId);
            sessionStorage.removeItem('streamlit_session_id');
            this.sessionId = null;
        } catch (error) {
            console.error('Error cleaning up session:', error);
        }
    }
}

if (!window.streamlitSessionManager) {
    window.streamlitSessionManager = new StreamlitSessionManager();
}
</script>
"""

# Inject the JavaScript
st.markdown(session_js, unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>💰 Z Analyzer</h1>
    <h3 style='color: #ffffff; opacity: 0.9; font-weight: 300;'>Session-Based Financial Document Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Session Management")

    # Display current session info
    if st.session_state.session_id:
        session_info = get_session_info(st.session_state.session_id)
        if session_info:
            st.markdown(f"""
            <div class="session-info">
                <h4>Current Session</h4>
                <p><strong>ID:</strong> {st.session_state.session_id[:8]}...</p>
                <p><strong>Documents:</strong> {session_info.get('document_count', 0)}</p>
                <p><strong>Created:</strong> {session_info.get('created_at', 'Unknown')[:19]}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Session may have expired")
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #17a2b8, #6f42c1);
                    color: white; padding: 1rem; border-radius: 10px;
                    text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0; color: white;'>📤 Getting Started</h4>
            <p style='margin: 0; opacity: 0.9;'>Upload documents to begin</p>
        </div>
        """, unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more Report to analyze"
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            result = upload_files_to_api(uploaded_files)
            if result:
                st.session_state.session_id = result['session_id']
                st.session_state.documents_uploaded = True
                st.session_state.vector_store_ready = True
                st.session_state.chat_history = []
                st.session_state.uploaded_files = result['filenames']

                # Update JavaScript with new session ID
                st.markdown(f"""
                <script>
                if (window.streamlitSessionManager) {{
                    window.streamlitSessionManager.setSessionId('{result["session_id"]}');
                }}
                </script>
                """, unsafe_allow_html=True)

                st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
                st.success(f"📋 Session ID: {result['session_id'][:8]}...")
                st.rerun()

    # Display uploaded files with better styling
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Uploaded Documents")
        for i, file in enumerate(st.session_state.uploaded_files, 1):
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.1);
                        padding: 0.5rem; border-radius: 8px;
                        margin-bottom: 0.5rem; border-left: 4px solid #28a745;'>
                <strong style='color: #ffffff;'>{i}. {file}</strong>
            </div>
            """, unsafe_allow_html=True)

    # Session management
    if st.session_state.session_id:
        st.divider()
        st.subheader("🔧 Session Actions")

        if st.button("🗑️ Delete Session"):
            result = delete_session(st.session_state.session_id)
            if result:
                st.session_state.session_id = None
                st.session_state.documents_uploaded = False
                st.session_state.vector_store_ready = False
                st.session_state.chat_history = []
                st.session_state.uploaded_files = []
                st.success("Session deleted successfully!")
                st.rerun()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("ℹ️ Session Info"):
            info = get_session_info(st.session_state.session_id)
            if info:
                st.json(info)

# Main chat interface
if not st.session_state.documents_uploaded:
    # Welcome message when no documents are uploaded
    st.markdown("""
    <div class="upload-section">
        <h2>🚀 Welcome to your personal analyzer!</h2>
        <p>To get started:</p>
        <ol>
            <li>Upload your financial PDF documents using the sidebar</li>
            <li>Click "Process Documents" to create a new session</li>
            <li>Start asking questions about your documents</li>
        </ol>
        <p><strong>Session Features:</strong></p>
        <ul>
            <li>✅ Automatic session management</li>
            <li>✅ Isolated document processing per session</li>
            <li>✅ Automatic cleanup on tab inactivity (5 minutes)</li>
            <li>✅ Manual session deletion</li>
            <li>✅ Real-time chat with your documents</li>
            <li>🌐 Web search for external data</li>
            <li>💬 Conversation memory within session</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat interface
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message">
                        <strong>You:</strong><br>{message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message">
                        <strong>Assistant:</strong><br>{message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Query input
    with st.container():
        # Check if already processing
        is_processing = 'processing_query' in st.session_state

        # Create a form to handle Enter key properly
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_query = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="e.g., What is the revenue growth? Compare margins with competitors...",
                    disabled=is_processing
                )
            with col2:
                send_button = st.form_submit_button(
                    "Processing..." if is_processing else "Send",
                    type="primary",
                    use_container_width=True,
                    disabled=is_processing
                )

    # Process query - only when Send button is clicked and there's a query
    if send_button and user_query.strip():
        # Check if this message is already being processed (prevent duplicates)
        if 'processing_query' not in st.session_state:
            st.session_state.processing_query = True

            # Store the query to process
            query_to_process = user_query.strip()

            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": query_to_process})

            # Get response
            with st.spinner("Thinking..."):
                try:
                    # Send message to API
                    response = send_chat_message(st.session_state.session_id, query_to_process)
                    if response:
                        # Extract response text
                        response_text = response.get("response", "I couldn't generate a response. Please try again.")

                        # Add assistant message to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        st.error("Failed to get response from the API.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Chat error: {e}")

                finally:
                    # Clear processing flag
                    if 'processing_query' in st.session_state:
                        del st.session_state.processing_query

                    # Rerun to refresh the interface
                    st.rerun()

# Footer with metrics
if st.session_state.documents_uploaded:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Documents", len(st.session_state.uploaded_files))
    with col2:
        st.metric("Messages", len(st.session_state.chat_history))
    with col3:
        st.metric("Session", st.session_state.session_id[:8] if st.session_state.session_id else "None")
    with col4:
        st.metric("Status", "🟢 Ready")

# Footer with session management info
st.markdown("""
---
<div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
    <p>💡 <strong>Session Management:</strong> Your session will automatically cleanup after 5 minutes of tab inactivity.</p>
    <p>🔄 Use the "Delete Session" button or upload new documents to start fresh.</p>
</div>
""", unsafe_allow_html=True)

# Instructions at bottom
with st.expander("ℹ️ How to use this analyzer"):
    st.markdown("""
    ### Quick Start Guide

    **1. Upload Documents**
    - Click on the file uploader in the sidebar
    - Select one or more PDF files
    - Click "Process Documents"

    **2. Ask Questions**
    - Type your question in the text input
    - Press Enter or click Send
    - The bot will search your documents and provide answers

    **3. Example Questions**
    - "What is the revenue for Q4?"
    - "Show me the operating margins"
    - "Compare our performance with Apple"
    - "Calculate the year-over-year growth"
    - "What factors affected profitability?"

    **4. Session Features**
    - 📄 Multi-document support per session
    - 🔍 Intelligent document search
    - 🌐 Web search for external data
    - 💬 Conversation memory within session
    - 📊 Financial calculations
    - ⏰ Automatic cleanup after 5 minutes inactivity
    - 🗑️ Manual session deletion

    **5. Tips**
    - Be specific with your questions
    - Each session isolates your documents and conversations
    - Sessions auto-delete after 5 minutes of tab inactivity
    - Use "Delete Session" to manually clean up
    - Upload new documents to create a fresh session
    """)
