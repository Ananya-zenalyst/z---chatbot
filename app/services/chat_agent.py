from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.services.vector_store import VectorStoreService
from app.utils.tools import web_search_tool
from app.core.config import settings
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory dictionary to store conversation histories per session_id
# For production, this should be replaced with a persistent store like Redis or a database.
CHAT_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}

class ChatAgent:
    """
    The main chat agent that orchestrates the response generation process.
    Enhanced with user-friendly conclusions and better explanations.
    """

    def __init__(self):
        # 1. Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )

        # 2. Define the agent's prompt template with enhanced conclusion capability
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a highly skilled financial analyst assistant specialized in providing accurate, comprehensive, and user-friendly responses.

YOUR PRIMARY GOALS:
1. Extract and use ALL relevant information from documents
2. Provide complete and accurate answers
3. Make complex financial data easy to understand
4. Always conclude with clear, actionable insights

OPERATIONAL WORKFLOW:

STEP 1 - DOCUMENT SEARCH (MANDATORY):
- ALWAYS start by using 'Document_Retriever' to search the uploaded PDFs
- Search comprehensively - don't stop at the first result
- Look for ALL relevant data points, not just the primary metric
- Pay attention to context, time periods, and related information

STEP 2 - INFORMATION SYNTHESIS:
- Combine information from multiple sources/pages
- Identify patterns, trends, and relationships
- Calculate derived metrics if needed (percentages, growth rates, etc.)
- Note any important contextual factors

STEP 3 - WEB SEARCH (ONLY IF NEEDED):
- Use 'Web_Search' only for information not in PDFs
- Ensure temporal consistency (same time periods)
- Clearly distinguish between document and web sources

RESPONSE STRUCTURE:

1. DIRECT ANSWER:
   Start with the specific answer to the question
   Include the main numbers or facts requested

2. DETAILED EXPLANATION:
   Break down complex information into digestible parts
   Show calculations step-by-step when applicable
   Provide context for better understanding

3. SUPPORTING DETAILS:
   Include relevant additional information
   Show trends or comparisons
   Mention factors affecting the metrics

4. USER-FRIENDLY CONCLUSION:
   Summarize the key takeaways in simple terms
   Explain what this means for the business
   Provide actionable insights or implications
   Use analogies or examples when helpful

5. SOURCE ATTRIBUTION:
   Cite specific documents and page numbers
   Distinguish between PDF and web sources

EXAMPLE RESPONSE FORMAT:

Question: "What is our gross margin?"

Answer:
**Gross Margin: 43.5% (Q3 2023)**

**Details:**
- Revenue: $15.2 billion
- Cost of Goods Sold: $8.6 billion
- Gross Profit: $6.6 billion
- Calculation: ($6.6B / $15.2B) × 100 = 43.5%

**Trend Analysis:**
- Q2 2023: 41.2%
- Q3 2023: 43.5% (↑ 2.3 percentage points)
- This represents a significant quarter-over-quarter improvement

**Key Drivers:**
- Improved pricing strategy (+1.5%)
- Reduced material costs (+0.8%)

**What This Means:**
Your gross margin of 43.5% means that for every dollar of sales, you keep about 44 cents after covering direct product costs. This is excellent performance - you're operating more efficiently than most competitors (industry average: 38-40%). The upward trend suggests your cost management and pricing strategies are working well.

**Recommendation:**
Continue monitoring material costs and consider locking in favorable supplier contracts while margins are strong.

*Source: Financial_Report_Q3_2023.pdf (Pages 12-14)*

CRITICAL RULES:

1. COMPLETENESS: Never give partial answers. Search thoroughly before responding.

2. ACCURACY: Use exact numbers from documents. Double-check calculations.

3. CLARITY: Explain financial jargon. Use simple language for conclusions.

4. CONCLUSIONS: ALWAYS end with a user-friendly summary that explains:
   - What the numbers mean in practical terms
   - Whether this is good/bad/neutral
   - What actions might be considered
   - How this compares to expectations or benchmarks

5. CONSISTENCY: Ensure all data points are from the same time period unless comparing across periods.

Remember: Your job is not just to report numbers, but to help users understand what those numbers mean for their business and what they should do about it."""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 3. Create the tools for the agent
        retriever = VectorStoreService.get_retriever()
        if retriever is None:
            raise RuntimeError("Vector store is not initialized. Cannot create retriever tool.")

        def retriever_func(query: str) -> str:
            """Enhanced wrapper function for the retriever with exact value preservation."""
            try:
                # Get documents with improved search
                docs = retriever.invoke(query)

                # If not enough results, try broader search
                if len(docs) < 3:
                    # Try searching for individual keywords
                    keywords = query.split()
                    for keyword in keywords:
                        if len(keyword) > 3:  # Skip short words
                            additional_docs = retriever.invoke(keyword)
                            docs.extend(additional_docs)

                    # Remove duplicates
                    seen = set()
                    unique_docs = []
                    for doc in docs:
                        doc_id = f"{doc.metadata.get('source', '')}/{doc.metadata.get('page', '')}{doc.page_content[:100]}"
                        if doc_id not in seen:
                            seen.add(doc_id)
                            unique_docs.append(doc)
                    docs = unique_docs[:15]  # Increased limit for better coverage

                if not docs:
                    return "No relevant information found in the uploaded documents. Please ensure the documents contain the information you're looking for."

                # Format the results with enhanced metadata
                result_parts = ["📊 DOCUMENT SEARCH RESULTS WITH EXACT VALUES:\n"]
                sources = {}

                for doc in docs:
                    source_name = doc.metadata.get('source', 'Unknown source')
                    page_num = doc.metadata.get('page', '')
                    source_location = doc.metadata.get('source_location', '')

                    # Extract exact values from metadata
                    financial_values = doc.metadata.get('chunk_financial_values', [])
                    dates = doc.metadata.get('chunk_dates', [])
                    percentages = doc.metadata.get('chunk_percentages', [])

                    # Create enhanced source key
                    source_key = f"{source_name}"
                    if source_location:
                        source_key += f" ({source_location})"
                    elif page_num:
                        source_key += f" (Page {page_num})"

                    if source_key not in sources:
                        sources[source_key] = {
                            'content': [],
                            'values': set(),
                            'dates': set(),
                            'percentages': set()
                        }

                    sources[source_key]['content'].append(doc.page_content)
                    sources[source_key]['values'].update(financial_values)
                    sources[source_key]['dates'].update(dates)
                    sources[source_key]['percentages'].update(percentages)

                # Build organized result with exact values highlighted
                for source_key, data in sources.items():
                    result_parts.append(f"\n📍 SOURCE: {source_key}")

                    # Show exact values found
                    if data['values']:
                        result_parts.append(f"💰 EXACT VALUES: {', '.join(data['values'])}")
                    if data['dates']:
                        result_parts.append(f"📅 DATES/PERIODS: {', '.join(data['dates'])}")
                    if data['percentages']:
                        result_parts.append(f"📊 PERCENTAGES: {', '.join(data['percentages'])}")

                    result_parts.append("\n--- CONTENT ---")
                    combined_content = "\n".join(data['content'])
                    # Preserve more content for accuracy
                    if len(combined_content) > 3000:
                        combined_content = combined_content[:3000] + "...[truncated for length]"
                    result_parts.append(combined_content)

                result_parts.append(f"\n\n✅ SEARCH COMPLETE: Found {len(docs)} relevant sections across {len(sources)} source locations")
                result_parts.append("⚠️ All values above are EXACT as found in documents - use them without modification")

                return "\n".join(result_parts)

            except Exception as e:
                logger.error(f"Error in document retrieval: {e}")
                return f"Error retrieving documents: {str(e)}"

        document_retriever_tool = Tool(
            name="Document_Retriever",
            func=retriever_func,
            description="Searches uploaded PDF documents for relevant financial information. ALWAYS use this FIRST before any analysis. Returns relevant content with source citations including page numbers."
        )

        self.tools = [document_retriever_tool, web_search_tool]

        # 4. Create the core agent logic
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=4,  # Allow multiple searches if needed
            handle_parsing_errors=True
        )

        # 5. Wrap the agent executor with history management
        self.agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: self._get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Retrieves or creates a conversation memory for a given session ID.
        """
        if session_id not in CHAT_HISTORIES:
            CHAT_HISTORIES[session_id] = InMemoryChatMessageHistory()
            logger.info(f"Created new chat history for session: {session_id}")
        return CHAT_HISTORIES[session_id]

    async def get_response(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Gets a response from the agent for a given user input and session.

        Args:
            user_input (str): The user's question.
            session_id (str): The unique identifier for the conversation.

        Returns:
            Dict[str, Any]: A dictionary containing the agent's output.
        """
        logger.info(f"Processing query for session '{session_id}': {user_input}")

        try:
            # Add context hints for better responses
            enhanced_input = user_input

            # Add hints for common queries to ensure comprehensive responses
            if any(word in user_input.lower() for word in ['margin', 'profit', 'revenue', 'sales']):
                enhanced_input += "\n[Note: Provide complete information including calculations, trends, and business implications]"

            if any(word in user_input.lower() for word in ['compare', 'versus', 'vs', 'competition']):
                enhanced_input += "\n[Note: Include detailed comparisons and explain what the differences mean]"

            if any(word in user_input.lower() for word in ['why', 'how', 'explain']):
                enhanced_input += "\n[Note: Provide thorough explanations in simple terms with examples]"

            response = await self.agent_with_chat_history.ainvoke(
                {"input": enhanced_input},
                config={"configurable": {"session_id": session_id}},
            )

            # Keep only the last 5 message pairs in history to avoid token limits
            history = CHAT_HISTORIES.get(session_id)
            if history:
                messages = history.messages
                if len(messages) > 10:  # 5 pairs of human/assistant messages
                    CHAT_HISTORIES[session_id].messages = messages[-10:]

            # Post-process response to ensure it has a conclusion
            output = response.get("output", "")

            # Check if response has a conclusion, add one if missing
            conclusion_keywords = ['means', 'conclusion', 'summary', 'takeaway', 'implication', 'recommendation']
            has_conclusion = any(keyword in output.lower() for keyword in conclusion_keywords)

            if not has_conclusion and len(output) > 100:
                # Add a simple conclusion prompt
                output += "\n\n**In Summary:** The data has been provided above. Please review the specific numbers and trends for your analysis."

            return {"output": output}

        except Exception as e:
            logger.error(f"Error during agent invocation for session '{session_id}': {e}", exc_info=True)
            return {
                "output": "I apologize, but I encountered an error while processing your request. Please try again with a simpler question or ensure your documents are properly uploaded."
            }

# Singleton instance to be used by the FastAPI app
chat_agent_instance = None

def get_chat_agent():
    """Get or create the chat agent instance."""
    global chat_agent_instance
    if chat_agent_instance is None:
        chat_agent_instance = ChatAgent()
    return chat_agent_instance
