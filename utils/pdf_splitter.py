
import re
from openai import OpenAI
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional, Any


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file using pdfplumber with improved table handling."""
    print(f"Extracting text from: {pdf_path} using pdfplumber")
    text = ""
    tables_found = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) 
                
                tables = page.extract_tables()
                table_text = ""
                
                if tables:
                    tables_found += len(tables)
                    for table_idx, table in enumerate(tables):
                        table_text += f"\n--- TABLE {i+1}-{table_idx+1} ---\n"
                        
                        for row in table:
                            formatted_row = "\t".join([str(cell) if cell is not None else "" for cell in row])
                            table_text += formatted_row + "\n"
                        
                        table_text += f"--- END TABLE {i+1}-{table_idx+1} ---\n\n"
                
                if page_text:
                    text += page_text + "\n"
                if table_text:
                    text += table_text
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed page {i+1}/{len(pdf.pages)}")

        print(f"Successfully extracted ~{len(text)} characters including {tables_found} tables.")
        text = re.sub(r'\n{3,}', '\n\n', text).strip()  # Clean up excessive newlines but preserve paragraph breaks
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path} with pdfplumber: {e}")
        return ""


def chunk_text_with_tables(text: str, chunk_size: int = 1536) -> List[str]:
    """Splits text into chunks while preserving table structures."""
    print(f"Chunking text using table-aware RecursiveCharacterTextSplitter (chunk size: {chunk_size})...")

    
    separators = [
        "\n\n\n",              # Large paragraph breaks (highest priority)
        "\n\n",                # Normal paragraph breaks
        "\n",                  # Line breaks (inside paragraphs)
        ". ",                  # Sentence breaks
        ", ",                  # Clause breaks
        " ",                   # Word breaks (lowest priority)
        ""                     # Character breaks (if all else fails)
    ]
    
    table_markers = re.findall(r'--- TABLE \d+-\d+ ---.*?--- END TABLE \d+-\d+ ---', text, re.DOTALL)
    
   
    large_tables = []
    for table in table_markers:
        if len(table) > chunk_size * 0.8:  
            large_tables.append(table)
    
    protected_text = text
    placeholders = {}
    for i, table in enumerate(large_tables):
        placeholder = f"[TABLE_PLACEHOLDER_{i}]"
        placeholders[placeholder] = table
        protected_text = protected_text.replace(table, placeholder)
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.20),  
        length_function=len,
    )
    
    chunks = text_splitter.split_text(protected_text)
    
    table_chunks = []
    for table in large_tables:
        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 3,  
            chunk_overlap=int(chunk_size * 0.20),  
            length_function=len,
        )
        table_chunks.extend(table_splitter.split_text(table))
    
    final_chunks = []
    for chunk in chunks:
        has_placeholder = False
        for placeholder, table in placeholders.items():
            if placeholder in chunk:
                has_placeholder = True
                if len(table) <= chunk_size:
                    chunk = chunk.replace(placeholder, table)
                else:
                    chunk = chunk.replace(placeholder, 
                                         "[Table was too large and is processed separately]")
        
        if chunk.strip():  
            final_chunks.append(chunk)
    
    #final_chunks.extend(table_chunks)
    
    print(f"Text split into {len(final_chunks)} chunks, including special handling for tables.")
    return final_chunks


