import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging
import re
from collections import Counter

# Configure logging to see the progress and any potential issues.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Enhanced document processor with better text extraction, table handling,
    and intelligent chunking for financial documents.
    """

    def __init__(self, chunk_size: int = 1600, chunk_overlap: int = 800):
        """
        Initializes the DocumentProcessor with optimized settings for financial documents.

        Args:
            chunk_size (int): Reduced chunk size for better granularity
            chunk_overlap (int): Increased overlap to maintain context
        """
        # Store chunk size for later reference
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use smaller chunks with more overlap for better information retention
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n\n",  # Multiple line breaks (section boundaries)
                "\n\n",    # Paragraph boundaries
                "\n",      # Line boundaries
                ". ",      # Sentence boundaries
                "; ",      # Clause boundaries
                ", ",      # Phrase boundaries
                " ",       # Word boundaries
                ""         # Character boundaries  
                
            ]
        )

        # Financial patterns for enhanced extraction
        self.financial_patterns = {
            'currency': re.compile(r'\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|M|B))?', re.IGNORECASE),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'date': re.compile(r'(?:Q[1-4]\s*)?(?:20\d{2}|FY\s*20\d{2})', re.IGNORECASE),
            'metric': re.compile(r'(?:revenue|income|profit|margin|EBITDA|earnings|sales|growth|expense|cost)', re.IGNORECASE)
        }

    def extract_tables_from_page(self, page) -> str:
        """
        Safely extract tables from a PDF page with better formatting.
        """
        try:
            # Check if the page has the find_tables method (newer PyMuPDF versions)
            if hasattr(page, 'find_tables'):
                tabs = page.find_tables()
                if hasattr(tabs, 'tables') and tabs.tables:
                    table_text = "\n\n[TABLE DATA]:\n"
                    for table in tabs.tables:
                        # Extract table with pandas-like formatting
                        extracted = table.extract()
                        for row in extracted:
                            # Join cells with | separator for clarity
                            row_text = " | ".join([str(cell) if cell else "" for cell in row])
                            table_text += row_text + "\n"
                        table_text += "\n"
                    return table_text

            # Fallback: Try to detect tables using text blocks
            # This works with older PyMuPDF versions
            blocks = page.get_text("blocks")
            table_text = ""
            for block in blocks:
                if len(block) >= 5:  # blocks have at least 5 elements
                    text = block[4]
                    # Simple heuristic: if text has multiple tabs or pipes, it might be a table
                    if isinstance(text, str) and ('\t' in text or '|' in text):
                        table_text += f"\n[POTENTIAL TABLE]:\n{text}\n"

            return table_text

        except Exception as e:
            logger.debug(f"Table extraction not available or failed: {e}")
            return ""

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to improve structure and readability.
        """
        # Fix common PDF extraction issues
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single line breaks to spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers

        # Preserve important financial structures
        text = re.sub(r'(\$\s+)([\d,]+)', r'$\2', text)  # Fix split currency

        # Add section markers for better chunking
        text = re.sub(r'((?:Table of Contents|Executive Summary|Financial Highlights|'
                      r'Management Discussion|Risk Factors|Financial Statements|'
                      r'Notes to Financial Statements|Revenue|Income Statement|'
                      r'Balance Sheet|Cash Flow))', r'\n\n\n\1', text, flags=re.IGNORECASE)

        return text

    def extract_metadata_from_text(self, text: str, page_num: int) -> Dict[str, Any]:
        """
        Extract structured metadata from text content.
        """
        metadata = {"page": page_num}

        try:
            # Extract financial metrics
            currencies = self.financial_patterns['currency'].findall(text)
            if currencies:
                metadata['financial_values'] = currencies[:5]  # Top 5 values

            percentages = self.financial_patterns['percentage'].findall(text)
            if percentages:
                metadata['percentages'] = percentages[:5]

            dates = self.financial_patterns['date'].findall(text)
            if dates:
                metadata['time_periods'] = list(set(dates))[:3]

            # Identify section type
            text_lower = text[:500].lower() if len(text) > 500 else text.lower()
            if 'income statement' in text_lower or 'profit' in text_lower and 'loss' in text_lower:
                metadata['section'] = 'Income Statement'
            elif 'balance sheet' in text_lower or ('assets' in text_lower and 'liabilities' in text_lower):
                metadata['section'] = 'Balance Sheet'
            elif 'cash flow' in text_lower:
                metadata['section'] = 'Cash Flow'
            elif 'executive summary' in text_lower or 'overview' in text_lower:
                metadata['section'] = 'Executive Summary'

        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")

        return metadata
    

    async def process_documents(self, contents: List[bytes], filenames: List[str]) -> List[Document]:
        """
        Enhanced document processing with better extraction and chunking.
        """
        all_docs = []
        logger.info(f"Starting enhanced processing for {len(contents)} PDF file(s).")

        for file_content, filename in zip(contents, filenames):
            try:
                # Open the PDF document from the byte stream
                pdf_document = fitz.open(stream=file_content, filetype="pdf")
                logger.info(f"Processing '{filename}' with {len(pdf_document)} pages")

                # Collect all text first to create a summary
                full_text = ""
                page_texts = []

                # Process each page separately for better granularity
                for page_num, page in enumerate(pdf_document, 1):
                    try:
                        # Extract text from page
                        page_text = page.get_text("text")

                        if not page_text or len(page_text.strip()) < 10:
                            # Try alternative extraction method if text is empty
                            page_text = page.get_text("blocks")
                            if isinstance(page_text, list):
                                page_text = "\n".join([block[4] if len(block) > 4 and isinstance(block[4], str) else "" for block in page_text])

                        # Skip if still no meaningful text
                        if not page_text or len(page_text.strip()) < 10:
                            logger.debug(f"Skipping page {page_num} of '{filename}' - no meaningful text")
                            continue

                        # Extract tables separately (won't fail if not available)
                        table_text = self.extract_tables_from_page(page)

                        # Combine text and tables
                        full_page_text = f"[Page {page_num}]\n{page_text}"
                        if table_text:
                            full_page_text += f"\n{table_text}"

                        # Preprocess the text
                        processed_text = self.preprocess_text(full_page_text)

                        # Skip if processed text is too short
                        if len(processed_text.strip()) < 50:
                            logger.debug(f"Skipping page {page_num} after preprocessing - text too short")
                            continue

                        # Store for summary
                        page_texts.append(page_text)
                        full_text += page_text + "\n"

                        # Extract metadata for this page
                        page_metadata = self.extract_metadata_from_text(processed_text, page_num)
                        page_metadata['source'] = filename
                        page_metadata['total_pages'] = len(pdf_document)

                        # Create document for this page
                        page_doc = Document(
                            page_content=processed_text,
                            metadata=page_metadata
                        )

                        # Split into chunks if needed
                        if len(processed_text) > self.chunk_size:
                            chunks = self.text_splitter.split_documents([page_doc])
                            # Preserve page metadata in chunks
                            for chunk in chunks:
                                chunk.metadata.update(page_metadata)
                            all_docs.extend(chunks)
                        else:
                            all_docs.append(page_doc)

                        logger.debug(f"Processed page {page_num} of '{filename}' successfully")

                    except Exception as e:
                        logger.warning(f"Error processing page {page_num} of '{filename}': {e}")
                        continue

                # Create a summary chunk with key information if we have text
                if full_text and len(full_text.strip()) > 100:
                    summary_text = self.create_document_summary(full_text, filename)
                    if summary_text:
                        summary_doc = Document(
                            page_content=summary_text,
                            metadata={
                                "source": filename,
                                "type": "document_summary",
                                "total_pages": len(pdf_document)
                            }
                        )
                        all_docs.append(summary_doc)

                pdf_document.close()

                # Log results for this document
                doc_chunks = [d for d in all_docs if d.metadata.get('source') == filename]
                if doc_chunks:
                    logger.info(f"Successfully processed '{filename}': {len(doc_chunks)} chunks created")
                else:
                    logger.warning(f"No chunks created for '{filename}' - document might be empty or unreadable")

            except Exception as e:
                logger.error(f"Error processing file '{filename}': {e}", exc_info=True)
                # Try a fallback simple extraction
                try:
                    pdf_document = fitz.open(stream=file_content, filetype="pdf")
                    simple_text = ""
                    for page in pdf_document:
                        simple_text += page.get_text() + "\n"

                    if simple_text and len(simple_text.strip()) > 100:
                        # Create at least one document with the raw text
                        fallback_doc = Document(
                            page_content=simple_text[:5000],  # Limit size
                            metadata={
                                "source": filename,
                                "type": "fallback_extraction",
                                "error": str(e)
                            }
                        )
                        all_docs.append(fallback_doc)
                        logger.info(f"Used fallback extraction for '{filename}'")
                    pdf_document.close()
                except Exception as fallback_error:
                    logger.error(f"Fallback extraction also failed for '{filename}': {fallback_error}")
                continue

        if not all_docs:
            logger.warning("No documents could be processed. Please check if the PDFs contain readable text.")
        else:
            logger.info(f"Total chunks created from all PDFs: {len(all_docs)}")

        return all_docs

    def create_document_summary(self, full_text: str, filename: str) -> str:
        """
        Create a summary chunk with key financial information.
        """
        try:
            summary_parts = [f"DOCUMENT SUMMARY: {filename}\n"]

            # Extract key financial metrics
            revenues = self.financial_patterns['currency'].findall(full_text)
            if revenues:
                # Filter for likely revenue figures (larger amounts)
                significant_amounts = [r for r in revenues if 'billion' in r.lower() or 'million' in r.lower()][:5]
                if significant_amounts:
                    summary_parts.append(f"Key Financial Figures: {', '.join(significant_amounts)}")

            # Extract time periods
            dates = self.financial_patterns['date'].findall(full_text)
            if dates:
                unique_dates = list(set(dates))[:5]
                summary_parts.append(f"Time Periods Covered: {', '.join(unique_dates)}")

            # Extract percentages (likely important metrics)
            percentages = self.financial_patterns['percentage'].findall(full_text)
            if percentages:
                unique_percentages = list(set(percentages))[:10]
                summary_parts.append(f"Key Percentages: {', '.join(unique_percentages)}")

            # Look for company names (capitalized words that appear frequently)
            words = re.findall(r'\b[A-Z][a-z]+\b', full_text)
            if words:
                word_counts = Counter(words)
                common_names = [word for word, count in word_counts.most_common(5) if count > 5 and len(word) > 3]
                if common_names:
                    summary_parts.append(f"Key Entities: {', '.join(common_names)}")

            return "\n".join(summary_parts) if len(summary_parts) > 1 else ""

        except Exception as e:
            logger.debug(f"Error creating summary: {e}")
            return ""