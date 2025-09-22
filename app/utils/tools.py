from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from app.core.config import settings
import logging
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

def format_web_search_results(raw_results):
    """
    Format web search results to include URLs and make them more readable.
    """
    try:
        # If raw_results is a string, try to parse it as JSON
        if isinstance(raw_results, str):
            try:
                results = json.loads(raw_results)
            except:
                return raw_results
        else:
            results = raw_results

        # Format the results with URLs prominently displayed
        formatted_output = []

        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content available')

                    # Truncate content if too long
                    if len(content) > 500:
                        content = content[:500] + "..."

                    formatted_output.append(
                        f"[{i}] {title}\n"
                        f"    URL: {url}\n"
                        f"    Content: {content}\n"
                    )

        if formatted_output:
            return "\n".join(formatted_output)
        else:
            return str(raw_results)
    except Exception as e:
        logger.error(f"Error formatting web search results: {e}")
        return str(raw_results)

# Initialize the web search tool using Tavily
try:
    # Set environment variable for Tavily
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

    # Create the base Tavily search tool
    tavily_search = TavilySearchResults(
        max_results=5,
        include_raw_content=False,
        include_answer=False
    )

    # Wrapper function that formats the results
    def enhanced_web_search(query: str) -> str:
        """
        Enhanced web search that returns formatted results with clear URLs.
        """
        try:
            raw_results = tavily_search.invoke(query)
            formatted_results = format_web_search_results(raw_results)

            # Add a header to make it clear these are web search results
            return (
                "=== WEB SEARCH RESULTS ===\n"
                f"Query: {query}\n\n"
                f"{formatted_results}\n"
                "=========================\n"
                "Note: Always cite the specific URL when using information from these sources."
            )
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Web search error: {str(e)}"

    # Create the tool with the enhanced function
    web_search_tool = Tool(
        name="Web_Search",
        func=enhanced_web_search,
        description=(
            "Search the web for financial information with TEMPORAL SPECIFICITY. "
            "ALWAYS include the time period (year/quarter) from the PDF documents in your search query. "
            "Example: 'Apple gross margin Q3 2023' not just 'Apple gross margin'. "
            "Returns results with source URLs that MUST be cited when using the information. "
            "Only use this when data is NOT available or derivable from PDFs. "
            "The tool returns structured results with titles, URLs, and content excerpts."
        )
    )

    logger.info("Enhanced web search tool initialized successfully.")

except Exception as e:
    logger.error(f"Failed to initialize web search tool: {e}")
    # Create a dummy tool that returns an error message
    def dummy_search(*args, **kwargs):
        return "Web search is currently unavailable. Please check your Tavily API key."

    web_search_tool = Tool(
        name="Web_Search",
        func=dummy_search,
        description="Web search tool (currently unavailable)"
    )