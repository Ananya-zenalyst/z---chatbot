import streamlit as st
import asyncio
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.chat_agent import get_session_chat_agent
from app.services.session_manager import session_manager
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Z Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Golden Yellow and Black theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        color: #FFD700;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%) !important;
        border-right: 2px solid #FFD700;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 6px 20px rgba(255,215,0,0.2);
        border: 1px solid #FFD700;
    }
    .chat-message.user {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        margin-left: 20%;
        font-weight: 500;
    }
    .chat-message.assistant {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        color: #FFD700;
        margin-right: 20%;
        border: 2px solid #FFD700;
    }
    .chat-message .message {
        width: 100%;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Welcome section */
    .upload-section {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255,215,0,0.3);
        border: 2px solid #FFD700;
        color: #FFD700;
    }
    .upload-section h2 {
        color: #FFD700 !important;
    }
    .upload-section p, .upload-section li {
        color: #FFD700;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(255,215,0,0.2);
        border: 1px solid #FFD700;
        color: #FFD700;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #FFD700 !important;
        border: 2px solid #FFD700 !important;
        border-radius: 10px;
        font-size: 16px;
        caret-color: #FFD700 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FFA500 !important;
        box-shadow: 0 0 10px rgba(255,215,0,0.3);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,215,0,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,215,0,0.5);
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
    }

    /* File uploader */
    .uploadedFile {
        background-color: #1a1a1a;
        border-radius: 10px;
        border: 2px solid #FFD700;
        color: #FFD700;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #FFD700;
        border-radius: 10px;
        background-color: rgba(255,215,0,0.05);
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: #1a1a1a;
        color: #90EE90;
        border: 2px solid #90EE90;
        border-radius: 10px;
    }
    .stError {
        background-color: #1a1a1a;
        color: #FF6B6B;
        border: 2px solid #FF6B6B;
        border-radius: 10px;
    }
    .stInfo {
        background-color: #1a1a1a;
        color: #87CEEB;
        border: 2px solid #87CEEB;
        border-radius: 10px;
    }
    .stWarning {
        background-color: #1a1a1a;
        color: #FFD700;
        border: 2px solid #FFD700;
        border-radius: 10px;
    }

    /* Title and headers */
    h1, h2, h3, h4 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }

    /* Sidebar text and headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #black !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border: 1px solid #FFD700;
        border-radius: 10px;
        color: #FFD700 !important;
    }
    .streamlit-expanderContent {
        background-color: #1a1a1a;
        border: 1px solid #FFD700;
        border-top: none;
        color: #FFD700;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border: 2px solid #FFD700;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(255,215,0,0.2);
    }
    [data-testid="metric-container"] label {
        color: #FFD700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFD700 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #FFD700 !important;
    }

    /* Container backgrounds */
    .block-container {
        background-color: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }

    /* Session info styling */
    .session-info {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(255,215,0,0.3);
    }
    .session-info h4 {
        color: #000000 !important;
        margin-bottom: 0.5rem;
    }
    .session-info p {
        color: #000000 !important;
        margin: 0.25rem 0;
    }

    /* Dividers */
    hr {
        border-color: #FFD700 !important;
        opacity: 0.5;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #FFD700;
        border: 2px solid #FFD700;
    }

    /* Additional golden touches */
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #FFD700, #FFA500);
    }
</style>
""", unsafe_allow_html=True)

# Direct service access (process-based approach, no API)

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

# Direct Service Functions (Process-based approach)
async def process_documents_locally(files):
    """Process documents directly using local services"""
    try:
        logger.info(f"Processing {len(files)} files locally.")

        # Create a new session
        session_id = session_manager.create_session()

        # Read file contents
        contents = [file.getvalue() for file in files]
        filenames = [file.name for file in files]

        # Process documents
        processor = DocumentProcessor()
        documents = await processor.process_documents(contents, filenames)

        if not documents:
            session_manager.delete_session(session_id)
            st.error("Could not extract any content from the provided files.")
            return None

        # Add documents to session
        session_manager.add_documents_to_session(session_id, documents)

        logger.info(f"Successfully processed {len(documents)} document chunks for session {session_id}.")
        return {
            "message": f"Successfully uploaded and processed {len(files)} files.",
            "session_id": session_id,
            "filenames": filenames
        }
    except Exception as e:
        logger.error(f"Error processing documents: {e}\n{traceback.format_exc()}")
        st.error(f"Processing error: {e}")
        return None

async def chat_locally(session_id, query):
    """Handle chat directly using local services"""
    try:
        # Check if session exists
        if not session_manager.session_exists(session_id):
            st.error(f"Session {session_id} not found. Please upload documents first.")
            return None

        # Get session-specific vector store
        session_vector_store = session_manager.get_session_vector_store(session_id)
        if not session_vector_store:
            st.error(f"No documents found for session {session_id}. Please upload documents first.")
            return None

        logger.info(f"Processing chat request for session_id: '{session_id}'")

        # Get chat agent with session-specific retriever
        chat_agent = get_session_chat_agent(session_vector_store)
        response = await chat_agent.get_response(query, session_id)

        return {
            "session_id": session_id,
            "response": response.get("output", "Sorry, I could not generate a response.")
        }
    except Exception as e:
        logger.error(f"Error during chat: {e}\n{traceback.format_exc()}")
        st.error(f"Chat error: {e}")
        return None

def delete_session_locally(session_id):
    """Delete session directly using local services"""
    try:
        if not session_manager.session_exists(session_id):
            st.error(f"Session {session_id} not found.")
            return None

        success = session_manager.delete_session(session_id)
        if success:
            return {
                "message": f"Session {session_id} successfully deleted.",
                "session_id": session_id
            }
        else:
            st.error(f"Failed to delete session {session_id}.")
            return None
    except Exception as e:
        logger.error(f"Error deleting session: {e}\n{traceback.format_exc()}")
        st.error(f"Deletion error: {e}")
        return None

def get_session_info_locally(session_id):
    """Get session information directly"""
    try:
        return session_manager.get_session_info(session_id)
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
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

# Session Management JavaScript (modified for local process)
session_js = """
<script>
class StreamlitSessionManager {
    constructor() {
        this.sessionId = null;
        this.isActive = true;
        this.inactivityTimer = null;
        this.INACTIVITY_TIMEOUT = 5 * 60 * 1000; // 5 minutes
        // No API_BASE_URL needed for process-based approach
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
        // For process-based approach, we'll rely on Streamlit's session state
        // The actual cleanup will be handled server-side when the process ends
        console.log('Session marked for cleanup:', this.sessionId);
        sessionStorage.removeItem('streamlit_session_id');
        this.sessionId = null;
    }
}

if (!window.streamlitSessionManager) {
    window.streamlitSessionManager = new StreamlitSessionManager();
}
</script>
"""

# Inject the JavaScript
st.markdown(session_js, unsafe_allow_html=True)

# Header with enhanced golden styling
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(255,215,0,0.1) 0%, rgba(255,165,0,0.1) 100%); border-radius: 20px; margin-bottom: 2rem; border: 2px solid #FFD700;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; color: #FFD700; text-shadow: 3px 3px 6px rgba(0,0,0,0.9);'>Z Analyzer</h1>
    <h3 style='color: #FFA500; opacity: 0.95; font-weight: 400; letter-spacing: 1px;'>Premium Financial Document Analysis System</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Session Management")

    # Display current session info
    if st.session_state.session_id:
        session_info = get_session_info_locally(st.session_state.session_id)
        if session_info:
            # Format the datetime properly
            created_at = session_info.get('created_at', 'Unknown')
            if created_at != 'Unknown':
                created_at_str = created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(created_at, 'strftime') else str(created_at)[:19]
            else:
                created_at_str = 'Unknown'

            st.markdown(f"""
            <div class="session-info">
                <h4>Current Session</h4>
                <p><strong>ID:</strong> {st.session_state.session_id[:8]}...</p>
                <p><strong>Documents:</strong> {session_info.get('document_count', 0)}</p>
                <p><strong>Created:</strong> {created_at_str}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Session may have expired")
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FFD700, #FFA500);
                    color: #000000; padding: 1rem; border-radius: 10px;
                    text-align: center; margin-bottom: 1rem; font-weight: 500;
                    box-shadow: 0 4px 15px rgba(255,215,0,0.3);'>
            <h4 style='margin: 0; color: #000000;'> Getting Started</h4>
            <p style='margin: 0; opacity: 0.9; color: black;'>Upload documents to begin</p>
        </div>
        """, unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Document files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more Report to analyze"
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            # Run async function in sync context
            result = asyncio.run(process_documents_locally(uploaded_files))
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

                st.success(f"Successfully processed {len(uploaded_files)} files!")
                st.success(f" Session ID: {result['session_id'][:8]}...")
                st.rerun()

    # Display uploaded files with better styling
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Uploaded Documents")
        for i, file in enumerate(st.session_state.uploaded_files, 1):
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, rgba(255,215,0,0.2) 0%, rgba(255,165,0,0.1) 100%);
                        padding: 0.5rem; border-radius: 8px;
                        margin-bottom: 0.5rem; border-left: 4px solid #000;'>
                <strong style='color: #FFD700;'>{i}. {file}</strong>
            </div>
            """, unsafe_allow_html=True)

    # Session management
    if st.session_state.session_id:
        st.divider()
        st.subheader(" Session Actions")

        if st.button("Delete Session"):
            result = delete_session_locally(st.session_state.session_id)
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

        if st.button("Session Info"):
            info = get_session_info_locally(st.session_state.session_id)
            if info:
                st.json(info)

# Main chat interface
if not st.session_state.documents_uploaded:
    # Welcome message when no documents are uploaded
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: #FFD700;"> Welcome to Your Premium Financial Analyzer!</h2>
        <p style="color: #FFA500; font-size: 1.1rem;">To get started:</p>
        <ol style="color: #FFD700;">
            <li>Upload your financial documents using the sidebar</li>
            <li>Click "Process Documents" to create a new session</li>
            <li>Start asking questions about your documents</li>
        </ol>
        <p style="color: #FFA500; font-size: 1.1rem; margin-top: 1.5rem;"><strong>Premium Features:</strong></p>
        <ul style="color: #FFD700;">
            <li> Automatic session management</li>
            <li>Isolated document processing per session</li>
            <li>Automatic cleanup on tab inactivity (5 minutes)</li>
            <li>Manual session deletion</li>
            <li>Real-time chat with your documents</li>
            <li>Web search for external data</li>
            <li>Conversation memory within session</li>
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
                        <strong style="color: #FFD700;">Z Analyzer:</strong><br>{message["content"]}
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
                    # Process message directly using local services
                    response = asyncio.run(chat_locally(st.session_state.session_id, query_to_process))
                    if response:
                        # Extract response text
                        response_text = response.get("response", "I couldn't generate a response. Please try again.")

                        # Add assistant message to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        st.error("Failed to get response from the chat agent.")

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
        st.metric("Status", "Ready")

# Footer with session management info
st.markdown("""
---
<div style="text-align: center; font-size: 0.9rem; background: linear-gradient(135deg, rgba(255,215,0,0.1) 0%, rgba(255,165,0,0.1) 100%); padding: 1rem; border-radius: 10px; border: 1px solid #FFD700; margin-top: 2rem;">
    <p style="color: #FFD700; margin: 0.5rem;"><strong>Session Management:</strong> Your session will automatically cleanup after 5 minutes of tab inactivity.</p>
    <p style="color: #FFA500; margin: 0.5rem;">Use the "Delete Session" button or upload new documents to start fresh.</p>
</div>
""", unsafe_allow_html=True)

# Instructions at bottom
with st.expander("How to use this analyzer"):
    st.markdown("""
    <div style="color: #FFD700;">
    <h3 style="color: #FFD700;">Quick Start Guide</h3>

    <h4 style="color: #FFA500;">1. Upload Documents</h4>
    <ul style="color: #FFD700;">
        <li>Click on the file uploader in the sidebar</li>
        <li>Select one or more files</li>
        <li>Click "Process Documents"</li>
    </ul>

    <h4 style="color: #FFA500;">2. Ask Questions</h4>
    <ul style="color: #FFD700;">
        <li>Type your question in the text input</li>
        <li>Press Enter or click Send</li>
        <li>The bot will search your documents and provide answers</li>
    </ul>

    <h4 style="color: #FFA500;">3. Example Questions</h4>
    <ul style="color: #FFD700;">
        <li>"What is the revenue for Q4?"</li>
        <li>"Show me the operating margins"</li>
        <li>"Compare our performance with Apple"</li>
        <li>"Calculate the year-over-year growth"</li>
        <li>"What factors affected profitability?"</li>
    </ul>

    <h4 style="color: #FFA500;">4. Session Features</h4>
    <ul style="color: #FFD700;">
        <li> Multi-document support per session</li>
        <li> Intelligent document search</li>
        <li> Web search for external data</li>
        <li> Conversation memory within session</li>
        <li> Financial calculations</li>
        <li> Automatic cleanup after 5 minutes inactivity</li>
        <li> Manual session deletion</li>
    </ul> 

    <h4 style="color: #FFA500;">5. Tips</h4>
    <ul style="color: #FFD700;">
        <li>Be specific with your questions</li>
        <li>Each session isolates your documents and conversations</li>
        <li>Sessions auto-delete after 5 minutes of tab inactivity</li>
        <li>Use "Delete Session" to manually clean up</li>
        <li>Upload new documents to create a fresh session</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
