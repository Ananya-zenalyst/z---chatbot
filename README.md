# 🤖 AI Financial Document Chatbot

An advanced AI-powered financial analysis chatbot that performs calculations, provides insights, and cites sources from both uploaded documents and real-time web data.

## ✨ Key Features

- **📄 Smart Document Processing**: Upload and analyze PDF financial documents with vector search
- **🧮 Advanced Calculations**: Performs financial calculations with final outcomes and explanations
- **🔍 Source Attribution**: Clear citations from both document sources and web searches
- **🌐 Real-time Web Data**: Integration with Tavily API for current market information
- **💬 Intelligent Chat**: Natural language queries with context-aware responses
- **🎨 Modern UI**: Beautiful Streamlit interface with gradient design
- **📊 Business Insights**: Actionable conclusions and impact analysis

## 🏗️ Architecture

- **AI Engine**: OpenAI GPT-4o with LangChain orchestration
- **Vector Database**: FAISS for semantic document search
- **Web Search**: Tavily API with source URL attribution
- **Document Processing**: PyMuPDF for PDF text extraction
- **Frontend**: Streamlit with custom CSS styling
- **Backend API**: FastAPI with async support

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Run the Application
```bash
# Streamlit UI (Recommended)
streamlit run streamlit_app.py

# FastAPI Backend (Optional)
uvicorn app.main:app --reload
```

### 4. Access Your Chatbot
- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## 💡 How to Use

1. **📤 Upload**: Drag and drop PDF financial documents
2. **⚡ Process**: Click "Process Documents" to create vector embeddings
3. **❓ Ask**: Type financial questions in natural language
4. **📊 Analyze**: Get calculated results with source citations

### Example Queries
- "Calculate the year-over-year revenue growth percentage"
- "Compare our operating margins with industry competitors"
- "What's the current stock price and how does it compare to Apple?"
- "Analyze the impact of cost reduction initiatives on profitability"

## 🎯 What Makes It Special

### 🧮 **Intelligent Calculations**
- Performs mathematical analysis automatically
- Explains business impact of financial metrics
- Provides actionable insights and conclusions

### 📚 **Source Attribution**
- **Document Sources**: `Source: [filename.pdf]`
- **Web Sources**: `Source: Web search via Tavily API` with specific sites
- **Citations**: Yahoo Finance, SEC filings, Reuters, etc.

### 🎨 **Professional Interface**
- Gradient background design
- High contrast for readability
- Mobile-responsive layout
- Real-time processing indicators

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/upload/` | POST | Upload and process PDFs |
| `/chat/` | POST | Chat with financial documents |

## 📁 Project Structure

```
chatbot/
├── app/
│   ├── core/config.py           # Environment configuration
│   ├── schemas/models.py        # Pydantic data models
│   ├── services/
│   │   ├── chat_agent.py        # AI chat logic with calculations
│   │   ├── document_processor.py # PDF processing
│   │   └── vector_store.py      # FAISS vector database
│   └── utils/tools.py           # Tavily web search tool
├── streamlit_app.py             # Modern Streamlit interface
├── requirements.txt             # Python dependencies
└── README.md                    # This documentation
```

## 🔧 Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API for GPT-4o and embeddings | ✅ |
| `TAVILY_API_KEY` | Tavily API for web search with sources | ✅ |

## 🌟 Advanced Features

- **Context Memory**: Remembers conversation history per session
- **Error Handling**: Graceful error recovery with user feedback
- **Duplicate Prevention**: Smart form handling prevents multiple submissions
- **Source Grouping**: Organizes document content by source file
- **Calculation Engine**: Automatic percentage, ratio, and growth calculations

## 🚀 Deployment Ready

The chatbot is optimized for deployment on:
- **Streamlit Cloud** (recommended)
- **Railway**
- **Render**
- **Heroku**
- **Docker containers**

Ready to push to GitHub and deploy globally! 🌍