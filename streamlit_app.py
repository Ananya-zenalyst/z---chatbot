import streamlit as st
import asyncio
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

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

# Header with enhanced styling
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>💰 Z Analyzer</h1>
    <h3 style='color: #ffffff; opacity: 0.9; font-weight: 300;'>Upload your financial Data and ask intelligent questions about them!</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")

    # Check vector store status with better styling
    if initialize_vector_store():
        st.markdown("""
        <div style='background: linear-gradient(135deg, #28a745, #20c997);
                    color: white; padding: 1rem; border-radius: 10px;
                    text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0; color: white;'>✅ Vector Store Ready</h4>
            <p style='margin: 0; opacity: 0.9;'>Data loaded and ready for queries</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.vector_store_ready = True
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #17a2b8, #6f42c1);
                    color: white; padding: 1rem; border-radius: 10px;
                    text-align: center; margin-bottom: 1rem;'>
            <h4 style='margin: 0; color: white;'>📤 Getting Started</h4>
            <p style='margin: 0; opacity: 0.9;'>Upload data to begin</p>
        </div>
        """, unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Here"
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more Report to analyze"
    )

    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                # Process uploaded files
                contents = []
                filenames = []

                # Show progress for multiple files
                progress_text = st.empty()
                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    progress_text.text(f"Reading file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / (len(uploaded_files) + 1))

                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()

                    # Validate PDF
                    if len(file_bytes) < 100:
                        st.warning(f"⚠️ File '{uploaded_file.name}' appears to be empty or too small")
                        continue

                    contents.append(file_bytes)
                    filenames.append(uploaded_file.name)

                if not contents:
                    st.error("No valid PDF files to process")
                    progress_text.empty()
                    progress_bar.empty()
                else:
                    progress_text.text("Processing PDF content...")
                    progress_bar.progress(0.5)

                    # Process documents
                    processor = DocumentProcessor()
                    documents = asyncio.run(processor.process_documents(contents, filenames))

                    if documents and len(documents) > 0:
                        progress_text.text("Creating vector store...")
                        progress_bar.progress(0.75)

                        # Load or create vector store
                        VectorStoreService.load_or_create_vector_store(documents)
                        st.session_state.vector_store_ready = True

                        # Only add successfully processed files
                        for filename in filenames:
                            if filename not in st.session_state.uploaded_files:
                                st.session_state.uploaded_files.append(filename)

                        progress_text.text("Complete!")
                        progress_bar.progress(1.0)

                        st.success(f"✅ Successfully processed {len(documents)} document chunks from {len(filenames)} file(s)")

                        # Show summary of what was extracted
                        with st.expander("📋 Processing Summary", expanded=False):
                            for filename in filenames:
                                file_chunks = [d for d in documents if d.metadata.get('source') == filename]
                                if file_chunks:
                                    st.write(f"**{filename}**: {len(file_chunks)} chunks")
                                    # Show sample metadata from first chunk
                                    if file_chunks[0].metadata.get('time_periods'):
                                        st.write(f"  - Time periods: {', '.join(file_chunks[0].metadata['time_periods'])}")
                                    if file_chunks[0].metadata.get('financial_values'):
                                        st.write(f"  - Sample values: {', '.join(file_chunks[0].metadata['financial_values'][:3])}")

                        st.rerun()
                    else:
                        progress_text.empty()
                        progress_bar.empty()
                        st.error("❌ Could not extract any content from the provided PDFs")
                        st.info("💡 Please ensure your PDFs:")
                        st.write("• Contain readable text (not scanned images)")
                        st.write("• Are not password protected")
                        st.write("• Are valid PDF files")
                        st.write("• Contain financial or business information")

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                logger.error(f"Document processing error: {e}", exc_info=True)

                # Provide helpful troubleshooting tips
                st.info("💡 Troubleshooting tips:")
                st.write("• Try uploading one file at a time to identify problematic PDFs")
                st.write("• Ensure PDFs are not corrupted or password-protected")
                st.write("• Check that files contain actual text, not just images")

                # Clean up progress indicators if they exist
                try:
                    progress_text.empty()
                    progress_bar.empty()
                except:
                    pass

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
    st.divider()
    st.subheader("🔧 Session Management")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if not st.session_state.vector_store_ready:
    # Welcome message when no documents are uploaded
    st.markdown("""
    <div class="upload-section">
        <h2> Welcome to your personal analyzer!</h2>
        <p>To get started:</p>
        <ol>
            <li>Upload your financial data using the sidebar</li>
            <li>Click "Process Data" to analyze them</li>
            <li>Ask questions</li>
        </ol>
        <p><strong>Features:</strong></p>
        <ul>
            <li>📊 Extract financial metrics and data</li>
            <li>🔍 Search through multiple documents</li>
            <li>🌐 Get real-time market data via web search</li>
            <li>💬 Maintain conversation context</li>
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
                    # Get or create chat agent
                    chat_agent = get_chat_agent()

                    # Get response
                    response = asyncio.run(
                        chat_agent.get_response(query_to_process, st.session_state.session_id)
                    )

                    # Extract response text
                    response_text = response.get("output", "I couldn't generate a response. Please try again.")

                    # Add assistant message to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

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
if st.session_state.vector_store_ready:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Documents", len(st.session_state.uploaded_files))
    with col2:
        st.metric("Messages", len(st.session_state.chat_history))
    with col3:
        st.metric("Session", st.session_state.session_id[:8])
    with col4:
        st.metric("Status", "🟢 Ready")

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

    **4. Features**
    - 📄 Multi-document support
    - 🔍 Intelligent document search
    - 🌐 Web search for external data
    - 💬 Conversation memory
    - 📊 Financial calculations

    **5. Tips**
    - Be specific with your questions
    - The bot remembers previous messages in the session
    - Click "New Session" to start fresh
    - Upload new documents anytime
    """)
