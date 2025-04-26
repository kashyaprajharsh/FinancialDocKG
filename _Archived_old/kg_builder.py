import os

import torch

# Import required libraries

torch.classes.__path__ = []
import re
import time
from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI
import pdfplumber
from dotenv import load_dotenv
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import datetime
import os
from google import genai
from pydantic import BaseModel, Field
import json
import instructor
from google.genai import types
import numpy as np
from sklearn.cluster import DBSCAN
from google import genai
from google.genai import types
import numpy as np
import networkx as nx
from pyvis.network import Network
from collections import Counter
import concurrent.futures
import math
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.graphs import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompts import qa_prompt, entity_prompt

import pandas as pd
from IPython.display import display


API_KEY = ""

# Initialize sentence transformer model with error handling
def init_sentence_transformer():
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Initialize the model with specific device
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        model.to(device)
        return model
    except Exception as e:
        print(f"Error initializing SentenceTransformer: {e}")
        print("Falling back to simpler model...")
        try:
            # Fallback to a simpler model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.to(device)
            return model
        except Exception as e:
            print(f"Critical error initializing SentenceTransformer: {e}")
            raise

generate_content_config = types.GenerateContentConfig(
        temperature=0.15,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
    )

client = genai.Client(api_key=API_KEY)
instructor_client = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS,config=generate_content_config)

# Initialize the sentence transformer model with error handling
try:
    sentence_model = init_sentence_transformer()
    print("Successfully initialized SentenceTransformer model")
except Exception as e:
    print(f"Failed to initialize SentenceTransformer: {e}")
    raise

# # Initialize the base client
# client = genai.Client(api_key=API_KEY)

# generation_config = genai.types.GenerationConfig(
# temperature=0.1, # Set desired temperature (0.0-1.0). 0.1 is low, for more deterministic output.
# response_mime_type="application/json", # Crucial for Gemini JSON mode
# top_p=0.95,
# )


# genai.configure(api_key=API_KEY)
        
# # Initialize Google's Gemini client
# instructor_client = instructor.from_gemini(
# client=genai.GenerativeModel(
#     model_name="models/gemini-2.5-flash-preview-04-17",
#     generation_config=generation_config
# ),
# mode=instructor.Mode.GEMINI_JSON)



def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file using pdfplumber with improved table handling."""
    print(f"Extracting text from: {pdf_path} using pdfplumber")
    text = ""
    tables_found = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract regular text
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) 
                
                # Extract tables and convert to text representation
                tables = page.extract_tables()
                table_text = ""
                
                if tables:
                    tables_found += len(tables)
                    for table_idx, table in enumerate(tables):
                        # Add table marker and metadata
                        table_text += f"\n--- TABLE {i+1}-{table_idx+1} ---\n"
                        
                        # Convert table to plain text with better formatting
                        for row in table:
                            # Replace None with empty string and join with tabs
                            formatted_row = "\t".join([str(cell) if cell is not None else "" for cell in row])
                            table_text += formatted_row + "\n"
                        
                        table_text += f"--- END TABLE {i+1}-{table_idx+1} ---\n\n"
                
                # Combine page text and table text
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



def chunk_text_with_tables(text: str, chunk_size: int = 2048) -> List[str]:
    """Splits text into chunks while preserving table structures."""
    print(f"Chunking text using table-aware RecursiveCharacterTextSplitter (chunk size: {chunk_size})...")

    # Define splitting logic that preserves tables
    # The recursive splitter tries these separators in order
    # We want tables to be kept intact, so we ensure table markers aren't used as split points
    separators = [
        "\n\n\n",              # Large paragraph breaks (highest priority)
        "\n\n",                # Normal paragraph breaks
        "\n",                  # Line breaks (inside paragraphs)
        ". ",                  # Sentence breaks
        ", ",                  # Clause breaks
        " ",                   # Word breaks (lowest priority)
        ""                     # Character breaks (if all else fails)
    ]
    
    # First identify table sections to protect them
    table_markers = re.findall(r'--- TABLE \d+-\d+ ---.*?--- END TABLE \d+-\d+ ---', text, re.DOTALL)
    
    # If there are very large tables, we might need to split them individually
    large_tables = []
    for table in table_markers:
        if len(table) > chunk_size * 0.8:  # If table is more than 80% of chunk size
            large_tables.append(table)
    
    # Replace large tables with placeholders
    protected_text = text
    placeholders = {}
    for i, table in enumerate(large_tables):
        placeholder = f"[TABLE_PLACEHOLDER_{i}]"
        placeholders[placeholder] = table
        protected_text = protected_text.replace(table, placeholder)
    
    # Configure the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.20),  # 10% overlap
        length_function=len,
    )
    
    # Split the text (with large tables as placeholders)
    chunks = text_splitter.split_text(protected_text)
    
    # Process large tables separately with higher chunk size to keep more structure
    table_chunks = []
    for table in large_tables:
        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 3,  # Use larger chunks for tables
            chunk_overlap=int(chunk_size * 0.20),  # More overlap for tables
            length_function=len,
        )
        table_chunks.extend(table_splitter.split_text(table))
    
    # Restore placeholders and add table chunks
    final_chunks = []
    for chunk in chunks:
        # Check if chunk contains any table placeholders
        has_placeholder = False
        for placeholder, table in placeholders.items():
            if placeholder in chunk:
                has_placeholder = True
                # Replace with original table content if small enough
                if len(table) <= chunk_size:
                    chunk = chunk.replace(placeholder, table)
                else:
                    # Remove placeholder to prevent confusion
                    chunk = chunk.replace(placeholder, 
                                         "[Table was too large and is processed separately]")
        
        if chunk.strip():  # Avoid empty chunks
            final_chunks.append(chunk)
    
    # Add the separate table chunks    final_chunks.extend(table_chunks)
    
    print(f"Text split into {len(final_chunks)} chunks, including special handling for tables.")
    return final_chunks



def visualize_chunks(chunks):
    print("--- Chunk Details ---")
    if chunks:
        # Convert string chunks to dictionary format with chunk numbers
        chunks_data = [{"chunk_number": i+1, "text": chunk} for i, chunk in enumerate(chunks)]
        
        # Create a DataFrame for better visualization
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df['word_count'] = chunks_df['text'].apply(lambda x: len(x.split()))
        
        # Display the DataFrame with selected columns
        display(chunks_df[['chunk_number', 'word_count', 'text']])
    else:
        print("No chunks were created (text might be shorter than chunk size).")
    print("-" * 25)

# Example usage:
# chunks = chunk_text_with_tables(your_text, chunk_size=525)
# Initialize lists to store results and failures
all_extracted_triples = []
failed_chunks = []




 #--- Knowledge Graph Schema ---
class Triple(BaseModel):
  """Represents a single knowledge graph triple (Subject-Predicate-Object)."""
  subject: str = Field(description="The entity or concept being described.")
  predicate: str = Field(description="The relationship or property connecting the subject and object.")
  object: str = Field(description="The entity, concept, or value related to the subject via the predicate.")

class TripleList(BaseModel):
    """A list of knowledge graph triples."""
    triples: List[Triple]



def process_chunk(chunk_idx: int, chunk_text: str, extraction_system_prompt: str, company_name: str) -> Tuple[List[dict], List[dict]]:
    """Process a single chunk of text to extract knowledge graph triples.
    
    Args:
        chunk_idx: Index of the current chunk
        chunk_text: Text content to process
        extraction_system_prompt: System prompt for the LLM
        extraction_user_prompt_template: Template for user prompt
        company_name: Name of the company being analyzed
        
    Returns:
        Tuple containing (successful_triples, failed_chunks)
    """
    print(f"\n--- Processing Chunk {chunk_idx + 1} --- ")
    
    if not chunk_text or not chunk_text.strip():
        print(f"Warning: Empty chunk received at index {chunk_idx}")
        return [], [{"chunk_idx": chunk_idx, "error": "Empty chunk", "chunk_text": chunk_text}]
    
    try:
        # 1. Format the User Prompt
        print("1. Formatting User Prompt...")
        system_prompt = extraction_system_prompt.format(
            text_chunk=chunk_text, 
            company_name=company_name
        )
        
        # 2. Make the API Call with retry logic
        print("2. Sending request to LLM...")
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = instructor_client.chat.completions.create(
                model="gemini-2.5-flash-preview-04-17",
                messages=[
                    {"role": "user", "content": system_prompt},
                ],
                response_model=TripleList,
                max_retries=3
                )
                
                # Convert response to list of dicts and add chunk information
                extracted_triples = []
                for triple in response.triples:
                    triple_dict = triple.dict()
                    triple_dict['chunk'] = chunk_idx  # Add chunk information
                    extracted_triples.append(triple_dict)
                
                if not extracted_triples:
                    print(f"Warning: No triples extracted from chunk {chunk_idx + 1}")
                    return [], [{"chunk_idx": chunk_idx, "error": "No triples extracted", "chunk_text": chunk_text}]
                
                print(f"Successfully extracted {len(extracted_triples)} triples from chunk {chunk_idx + 1}")
                return extracted_triples, []
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for chunk {chunk_idx + 1}. Error: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Failed to process chunk {chunk_idx + 1} after {max_retries} attempts. Error: {str(e)}"
                    print(error_msg)
                    return [], [{"chunk_idx": chunk_idx, "error": error_msg, "chunk_text": chunk_text}]
    
    except Exception as e:
        error_msg = f"Unexpected error processing chunk {chunk_idx + 1}: {str(e)}"
        print(error_msg)
        return [], [{"chunk_idx": chunk_idx, "error": error_msg, "chunk_text": chunk_text}]

def normalize_and_deduplicate_triples(all_extracted_triples: List[dict]) -> List[dict]:
    """
    Normalize and deduplicate knowledge graph triples.
    
    Args:
        all_extracted_triples: List of extracted triples
        
    Returns:
        List of normalized and deduplicated triples
    """
    # Initialize lists and tracking variables
    normalized_triples = []
    seen_triples = set()  # Tracks (subject, predicate, object) tuples
    original_count = len(all_extracted_triples)
    empty_removed_count = 0
    duplicates_removed_count = 0

    print(f"Starting normalization and de-duplication of {original_count} triples...")
    print("Processing triples for normalization (showing first 5 examples):")
    example_limit = 5
    processed_count = 0

    for i, triple in enumerate(all_extracted_triples):
        show_example = (i < example_limit)
        if show_example:
            print(f"\n--- Example {i+1} ---")
            print(f"Original Triple (Chunk {triple.get('chunk', '?')}): {triple}")
        
        subject_raw = triple.get('subject')
        predicate_raw = triple.get('predicate')
        object_raw = triple.get('object')
        chunk_num = triple.get('chunk', 'unknown')
        
        triple_valid = False
        normalized_sub, normalized_pred, normalized_obj = None, None, None

        if isinstance(subject_raw, str) and isinstance(predicate_raw, str) and isinstance(object_raw, str):
            # 1. Normalize
            normalized_sub = subject_raw.strip().lower()
            normalized_pred = re.sub(r'\s+', ' ', predicate_raw.strip().lower()).strip()
            normalized_obj = object_raw.strip().lower()
            if show_example:
                print(f"Normalized: SUB='{normalized_sub}', PRED='{normalized_pred}', OBJ='{normalized_obj}'")

            # 2. Filter Empty
            if normalized_sub and normalized_pred and normalized_obj:
                triple_identifier = (normalized_sub, normalized_pred, normalized_obj)
                
                # 3. De-duplicate
                if triple_identifier not in seen_triples:
                    normalized_triples.append({
                        'subject': normalized_sub,
                        'predicate': normalized_pred,
                        'object': normalized_obj,
                        'source_chunk': chunk_num
                    })
                    seen_triples.add(triple_identifier)
                    triple_valid = True
                    if show_example:
                        print("Status: Kept (New Unique Triple)")
                else:
                    duplicates_removed_count += 1
                    if show_example:
                        print("Status: Discarded (Duplicate)")
            else:
                empty_removed_count += 1
                if show_example:
                    print("Status: Discarded (Empty component after normalization)")
        else:
            empty_removed_count += 1  # Count non-string/missing as needing removal
            if show_example:
                print("Status: Discarded (Non-string or missing component)")
        processed_count += 1

    print(f"\n... Finished processing {processed_count} triples.")
    print(f"\nNormalization Summary:")
    print(f"Original triples: {original_count}")
    print(f"Empty/invalid triples removed: {empty_removed_count}")
    print(f"Duplicate triples removed: {duplicates_removed_count}")
    print(f"Final unique triples: {len(normalized_triples)}")
    
    return normalized_triples




def define_schema(triples: List[dict]) -> Dict[str, str]:
    """
    Define schema by generating definitions for each unique relation using LLM.
    
    Args:
        triples: List of knowledge graph triples
        
    Returns:
        Dictionary mapping relations to their definitions
    """
    print("\nPhase 2: Defining Schema...")
    
    # Extract unique predicates from triples
    unique_relations = set(triple['predicate'] for triple in triples)
    relation_definitions = {}
    
    print(f"Found {len(unique_relations)} unique relations")
    print("Generating definitions using Gemini...")

    for relation in unique_relations:
        try:
            # Use the same Gemini client we already have configured
            prompt = f"""Given the relation '{relation}' extracted from a company's financial documents,
            provide a clear, concise one-sentence definition of what this relation means in the context
            of business and financial reporting. Focus on precision and clarity."""
            
            response = instructor_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": prompt}],
                response_model=str,
                max_retries=3
            )
            
            relation_definitions[relation] = response.strip()
            print(f"✓ Defined: {relation}")
            
        except Exception as e:
            print(f"✗ Failed to define '{relation}': {str(e)}")
            relation_definitions[relation] = f"Undefined relation: {relation}"
    
    print(f"Schema definition complete. Defined {len(relation_definitions)} relations.")
    return relation_definitions


def get_embedding(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
    """
    Get embedding for a text using SentenceTransformer.
    
    Args:
        text: Text to embed
        task_type: Type of embedding task (ignored for SentenceTransformer)
        
    Returns:
        List of floats representing the embedding
    """
    try:
        # Convert text to string if it's not already
        text = str(text).strip()
        if not text:
            print("Warning: Empty text provided for embedding")
            return None
            
        # Get the device the model is on
        device = next(sentence_model.parameters()).device
        
        # Encode the text
        with torch.no_grad():  # Disable gradient calculation for inference
            embedding = sentence_model.encode(
                text,
                convert_to_tensor=True,
                device=device
            )
            # Convert to CPU and then to list
            return embedding.cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def get_embeddings_batch(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using SentenceTransformer.
    
    Args:
        texts: List of texts to embed
        task_type: Type of embedding task (ignored for SentenceTransformer)
        
    Returns:
        List of embedding vectors
    """
    try:
        # Clean and validate input texts
        texts = [str(text).strip() for text in texts]
        valid_texts = [text for text in texts if text]
        
        if not valid_texts:
            print("Warning: No valid texts provided for embedding")
            return []
            
        # Get the device the model is on
        device = next(sentence_model.parameters()).device
        
        # Batch encode texts
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings = sentence_model.encode(
                valid_texts,
                convert_to_tensor=True,
                device=device,
                batch_size=32  # Process in smaller batches to manage memory
            )
            # Convert to CPU and then to list
            return embeddings.cpu().numpy().tolist()
    except Exception as e:
        print(f"Error getting batch embeddings: {str(e)}")
        # Create zero vectors as fallback with the correct dimension
        embedding_dim = sentence_model.get_sentence_embedding_dimension()
        return [[0] * embedding_dim for _ in texts]

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    a = np.array(embedding1)
    b = np.array(embedding2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def canonicalize_schema(triples: List[dict], relation_definitions: Dict[str, str], 
                        similarity_threshold: float = 0.75) -> Tuple[List[dict], Dict[str, str]]:
    """
    Canonicalize schema by finding similar relations using embeddings.
    
    Args:
        triples: List of knowledge graph triples
        relation_definitions: Dictionary of relation definitions
        similarity_threshold: Threshold for similarity (default: 0.75)
        
    Returns:
        Tuple of (canonicalized triples, mapping of original to canonical relations)
    """
    print(f"\nPhase 3: Canonicalizing Schema (Similarity Threshold: {similarity_threshold})...")
    
    try:
        definitions = list(relation_definitions.values())
        relation_keys = list(relation_definitions.keys())
        
        # Get embeddings for all definitions (only need one type now)
        print("Generating embeddings...")
        embeddings = get_embeddings_batch(definitions, "")
        
        # Initialize canonical mapping
        canonical_map = {}
        processed = set()
        
        # Track similarity scores for analysis
        all_scores = []
        
        # Process each relation
        print("\nFinding similar relations...")
        for i, relation in enumerate(relation_keys):
            if relation in processed:
                continue
                
            similar_relations = []
            
            # Find similar relations
            for j, other_relation in enumerate(relation_keys):
                if i != j and other_relation not in processed:
                    # Only need one similarity score now
                    similarity = compute_similarity(embeddings[i], embeddings[j])
                    
                    # Track scores for analysis
                    all_scores.append({
                        'relation1': relation,
                        'relation2': other_relation,
                        'similarity': similarity
                    })
                    
                    # Check against threshold
                    if similarity > similarity_threshold:
                        similar_relations.append((other_relation, similarity))
            
            # If we found similar relations, create a group
            if similar_relations:
                # Sort by similarity score
                similar_relations.sort(key=lambda x: x[1], reverse=True)
                
                # Use the current relation as canonical form
                canonical_relation = relation
                processed.add(canonical_relation)
                canonical_map[canonical_relation] = canonical_relation
                
                # Map similar relations to the canonical form
                for similar_rel, score in similar_relations:
                    canonical_map[similar_rel] = canonical_relation
                    processed.add(similar_rel)
                    print(f"Mapped '{similar_rel}' → '{canonical_relation}' (similarity: {score:.2f})")
            else:
                # No similar relations found, use as its own canonical form
                canonical_map[relation] = relation
                processed.add(relation)
        
        # Update triples with canonical relations
        print("\nUpdating triples with canonical relations...")
        canonical_triples = []
        for triple in triples:
            canonical_triple = triple.copy()
            canonical_triple['predicate'] = canonical_map[triple['predicate']]
            canonical_triples.append(canonical_triple)
        
        # Summary statistics
        original_relations = len(set(t['predicate'] for t in triples))
        canonical_relations = len(set(t['predicate'] for t in canonical_triples))
        
        # Analyze similarity scores
        if all_scores:
            avg_similarity = sum(s['similarity'] for s in all_scores) / len(all_scores)
            print(f"\nSimilarity Analysis:")
            print(f"- Average similarity: {avg_similarity:.3f}")
            print(f"- Pairs above threshold: {len([s for s in all_scores if s['similarity'] > similarity_threshold])}")
        
        print(f"\nCanonicalization complete:")
        print(f"- Original unique relations: {original_relations}")
        print(f"- Canonical unique relations: {canonical_relations}")
        print(f"- Relations merged: {original_relations - canonical_relations}")
        
        return canonical_triples, canonical_map
        
    except Exception as e:
        print(f"Error during canonicalization: {str(e)}")
        return triples, {r: r for r in relation_definitions.keys()}


def build_and_visualize_knowledge_graph(company_name: str, triples_file: str = None, output_dir: str = None):
    """
    Build and visualize knowledge graph from the canonical triples using NetworkX.
    Provides multiple visualization options, graph analytics, and QA capabilities.
    """

    
    # Create company-specific directory structure
    company_dir = os.path.join("companies", company_name)
    if output_dir is None:
        output_dir = os.path.join(company_dir, "graph_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use company-specific triples file if not provided
    if triples_file is None:
        triples_file = os.path.join(company_dir, "canonical_triples.json")
    
    # Load the canonical triples
    print("Loading knowledge graph triples...")
    with open(triples_file, 'r') as f:
        triples = json.load(f)
    
    # Build the graph directly from JSON triples
    G = nx.DiGraph()
    for triple in triples:
        G.add_edge(
            triple['subject'],
            triple['object'],
            relation=triple['predicate']
        )
    
    # Save graph in GraphML format
    print("\nSaving graph in GraphML format...")
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"Saved GraphML file to: {graphml_path}")
    
    # Generate interactive HTML visualization using pyvis
    print("Generating interactive HTML visualization...")
    nt = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    nt.from_nx(G)
    nt.set_options('''
    {
        "nodes": {
            "shape": "dot",
            "size": 25,
            "font": {
                "size": 14
            }
        },
        "edges": {
            "font": {
                "size": 12
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -80000,
                "springLength": 250,
                "springConstant": 0.001
            }
        }
    }
    ''')
    html_path = os.path.join(output_dir, "knowledge_graph_interactive.html")
    try:
        nt.show(html_path)
    except Exception as e:
        print(f"HTML visualization failed: {e}, but GraphML file was created successfully")
    print(f"Saved interactive visualization to: {html_path}")
    
    # Initialize Graph QA capabilities
    print("\nInitializing Graph QA capabilities...")
    try:
        graph_qa = NetworkxEntityGraph(G)
        llm = ChatGoogleGenerativeAI(model_name="gemini-2.0-flash", api_key=API_KEY)
        qa_chain = GraphQAChain.from_llm(
            llm=llm,
            graph=graph_qa,
            qa_prompt=qa_prompt,
            entity_prompt=entity_prompt,
            verbose=True
        )
        print("Graph QA system initialized successfully!")
        return G, qa_chain, graphml_path
    except Exception as e:
        print(f"Error initializing Graph QA: {e}")
        return G, None, graphml_path

def process_chunk_batch(batch_chunks: List[str], start_idx: int, extraction_system_prompt: str, company_name: str) -> Tuple[List[dict], List[dict]]:
    """
    Process a batch of chunks in parallel.
    
    Args:
        batch_chunks: List of text chunks to process
        start_idx: Starting index of the batch
        extraction_system_prompt: System prompt for extraction
        company_name: Name of the company
    
    Returns:
        Tuple of (triples, failures) for the batch
    """
    batch_triples = []
    batch_failures = []
    
    for i, chunk in enumerate(batch_chunks):
        chunk_idx = start_idx + i
        try:
            # Pass the full chunks list but process only this chunk
            triples, failures = process_chunk(
                chunk_idx,
                chunk,  # Pass just this chunk instead of all batch_chunks
                extraction_system_prompt,
                company_name
            )
            batch_triples.extend(triples)
            batch_failures.extend(failures)
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            batch_failures.append({
                'chunk_number': chunk_idx,
                'error': str(e),
                'chunk': chunk[:100] + '...'  # Store first 100 chars of problematic chunk
            })
    
    return batch_triples, batch_failures

def parallel_process_chunks(chunks: List[str], extraction_system_prompt: str, company_name: str, 
                          batch_size: int = 10, max_workers: int = 4) -> Tuple[List[dict], List[dict]]:
    """
    Process chunks in parallel using batches.
    
    Args:
        chunks: List of all text chunks
        extraction_system_prompt: System prompt for extraction
        company_name: Name of the company
        batch_size: Number of chunks to process in each batch
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (all_triples, all_failures)
    """
    all_triples = []
    all_failures = []
    
    # Calculate number of batches
    num_chunks = len(chunks)
    num_batches = math.ceil(num_chunks / batch_size)
    
    print(f"\nProcessing {num_chunks} chunks in {num_batches} batches using {max_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create batch processing tasks
        future_to_batch = {}
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_chunks)
            batch_chunks = chunks[start_idx:end_idx]
            
            future = executor.submit(
                process_chunk_batch,
                batch_chunks,
                start_idx,
                extraction_system_prompt,
                company_name
            )
            future_to_batch[future] = (start_idx, end_idx)
        
        # Process completed batches
        completed_batches = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            start_idx, end_idx = future_to_batch[future]
            try:
                batch_triples, batch_failures = future.result()
                all_triples.extend(batch_triples)
                all_failures.extend(batch_failures)
                completed_batches += 1
                print(f"Completed batch {completed_batches}/{num_batches} (chunks {start_idx+1}-{end_idx})")
            except Exception as e:
                print(f"Batch {completed_batches + 1} failed: {str(e)}")
                all_failures.extend([{
                    'chunk_number': i + 1,
                    'error': f"Batch processing error: {str(e)}",
                    'chunk': 'Unknown'
                } for i in range(start_idx, end_idx)])
    
    return all_triples, all_failures

def parallel_schema_processing(normalized_triples: List[dict], batch_size: int = 50, max_workers: int = 4) -> Tuple[Dict[str, str], List[str]]:
    """
    Process schema definitions in parallel batches.
    
    Args:
        normalized_triples: List of normalized triples
        batch_size: Number of relations to process in each batch
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (relation_definitions, failed_relations)
    """
    # Extract unique relations
    unique_relations = list(set(triple['predicate'] for triple in normalized_triples))
    num_relations = len(unique_relations)
    num_batches = math.ceil(num_relations / batch_size)
    
    relation_definitions = {}
    failed_relations = []
    
    print(f"Processing {num_relations} unique relations in {num_batches} batches")
    
    def process_relation_batch(relations_batch):
        batch_definitions = {}
        batch_failures = []
        
        for relation in relations_batch:
            try:
                # Use the same Gemini client we already have configured
                prompt = f"""Given the relation '{relation}' extracted from a company's financial documents,
                provide a clear, concise one-sentence definition of what this relation means in the context
                of business and financial reporting. Focus on precision and clarity."""
                
                response = instructor_client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    response_model=str,
                    max_retries=3
                )
                
                batch_definitions[relation] = response.strip()
            except Exception as e:
                print(f"Failed to define relation '{relation}': {str(e)}")
                batch_failures.append(relation)
        
        return batch_definitions, batch_failures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {}
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_relations)
            relations_batch = unique_relations[start_idx:end_idx]
            
            future = executor.submit(process_relation_batch, relations_batch)
            future_to_batch[future] = batch_idx + 1
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                batch_definitions, batch_failures = future.result()
                relation_definitions.update(batch_definitions)
                failed_relations.extend(batch_failures)
                print(f"Completed relation batch {batch_num}/{num_batches}")
            except Exception as e:
                print(f"Relation batch {batch_num} failed: {str(e)}")
    
    return relation_definitions, failed_relations

def parallel_canonicalize_schema(triples: List[dict], relation_definitions: Dict[str, str],
                               similarity_threshold: float = 0.75,
                               batch_size: int = 10,
                               max_workers: int = 2):
    """
    Parallelized version of canonicalize schema function with rate limiting.
    """
    print(f"\nPhase 3: Parallel Canonicalizing Schema...")
    
    try:
        definitions = list(relation_definitions.values())
        relation_keys = list(relation_definitions.keys())
        
        # 1. Parallel embedding generation with smaller batches
        print("Generating embeddings in parallel...")
        embedding_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            def process_embeddings_batch(batch_defs):
                results = {}
                for definition in batch_defs:
                    try:
                        embedding = get_embedding(definition, "")
                        if embedding:
                            results[definition] = embedding
                    except Exception as e:
                        print(f"Error getting embedding: {str(e)}")
                return results
            
            # Submit tasks for embedding generation with smaller batches
            futures = []
            for i in range(0, len(definitions), batch_size):
                batch = definitions[i:i + batch_size]
                futures.append(executor.submit(process_embeddings_batch, batch))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    embedding_results.update(result)

        # 2. Parallel similarity computation
        print("Computing similarities in parallel...")
        similarity_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            def compute_batch_similarities(rel_batch):
                batch_results = []
                for rel1 in rel_batch:
                    for rel2 in relation_keys:
                        if rel1 != rel2:
                            try:
                                similarity = compute_similarity(
                                    embedding_results[relation_definitions[rel1]],
                                    embedding_results[relation_definitions[rel2]]
                                )
                                batch_results.append((rel1, rel2, similarity))
                            except Exception as e:
                                print(f"Error computing similarity: {str(e)}")
                return batch_results
            
            # Submit similarity computation tasks
            similarity_futures = []
            for i in range(0, len(relation_keys), batch_size):
                batch = relation_keys[i:i + batch_size]
                similarity_futures.append(executor.submit(compute_batch_similarities, batch))
            
            # Collect similarity results
            for future in concurrent.futures.as_completed(similarity_futures):
                similarity_results.extend(future.result())
        
        # 3. Sequential processing for canonical mapping (cannot be parallelized)
        print("Creating canonical mappings...")
        canonical_map = {}
        processed = set()
        
        for relation in relation_keys:
            if relation in processed:
                continue
                
            similar_relations = []
            for rel1, rel2, similarity in similarity_results:
                if rel1 == relation and rel2 not in processed:
                    if similarity > similarity_threshold:
                        similar_relations.append((rel2, similarity))
            
            if similar_relations:
                similar_relations.sort(key=lambda x: x[1], reverse=True)
                canonical_relation = relation
                processed.add(canonical_relation)
                canonical_map[canonical_relation] = canonical_relation
                
                for similar_rel, score in similar_relations:
                    canonical_map[similar_rel] = canonical_relation
                    processed.add(similar_rel)
                    print(f"Mapped '{similar_rel}' → '{canonical_relation}' (similarity: {score:.2f})")
            else:
                canonical_map[relation] = relation
                processed.add(relation)
        
        # 4. Parallel triple update with chunk preservation
        print("Updating triples with canonical relations...")
        canonical_triples = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            def update_triples_batch(batch_triples):
                return [{
                    'subject': t['subject'],
                    'predicate': canonical_map[t['predicate']],
                    'object': t['object'],
                    'source_chunk': t.get('source_chunk', 'unknown')  # Preserve chunk information
                } for t in batch_triples]
            
            # Process triple updates in parallel
            triple_futures = []
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                triple_futures.append(executor.submit(update_triples_batch, batch))
            
            # Collect updated triples
            for future in concurrent.futures.as_completed(triple_futures):
                canonical_triples.extend(future.result())
        
        # Summary statistics
        original_relations = len(set(t['predicate'] for t in triples))
        canonical_relations = len(set(t['predicate'] for t in canonical_triples))
        
        print(f"\nCanonicalization complete:")
        print(f"- Original unique relations: {original_relations}")
        print(f"- Canonical unique relations: {canonical_relations}")
        print(f"- Relations merged: {original_relations - canonical_relations}")
        
        return canonical_triples, canonical_map
        
    except Exception as e:
        print(f"Error during parallel canonicalization: {str(e)}")
        # Fallback to modified canonicalize_schema
        return canonicalize_schema(triples, relation_definitions, similarity_threshold)

def query_knowledge_graph(qa_chain, query: str) -> str:
    """
    Query the knowledge graph using the QA chain.
    
    Args:
        qa_chain: The initialized GraphQAChain
        query: The question to ask about the knowledge graph
        
    Returns:
        str: The answer to the query
    """
    if qa_chain is None:
        return "Error: Graph QA system is not initialized."
    
    try:
        print(f"\nProcessing query: {query}")
        response = qa_chain.invoke(query)
        print("\nResponse:", response)
        return response
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return error_msg

def main():
    """
    Main function to orchestrate the knowledge graph building process.
    """
    # Load environment variables if needed
    load_dotenv()
    
    # Import prompts from prompts.py
    from utils.prompts import extraction_system_prompt
    
    # Extract text from PDF
    pdf_path = "10K-filings/10K-filings/AAPL_10k.pdf"
    
    # Extract company name from PDF filename
    company_name = os.path.basename(pdf_path).split('_')[0]
    print(f"Processing 10-K filing for company: {company_name}")
    
    text = extract_text_from_pdf(pdf_path)
    
    # Chunk the text
    chunks = chunk_text_with_tables(text)
    
    # Visualize chunks
    visualize_chunks(chunks)
    
    # Process chunks in parallel
    print("\nProcessing chunks in parallel...")
    all_triples, all_failures = parallel_process_chunks(
            chunks, 
            extraction_system_prompt, 
            company_name,
            batch_size=10,
            max_workers=4
    )
    
    # Final summary of extraction
    print("\n=== Extraction Results ===")
    print(f"Total triples extracted: {len(all_triples)}")
    print(f"Total failed chunks: {len(all_failures)}")
    
    # Normalize and deduplicate triples
    print("\n=== Starting Normalization and Deduplication ===")
    normalized_triples = normalize_and_deduplicate_triples(all_triples)
    
    # Phase 2: Define schema in parallel
    print("\n=== Starting Parallel Schema Definition ===")
    relation_definitions, failed_relations = parallel_schema_processing(
        normalized_triples,
        batch_size=50,
        max_workers=4
    )
    
    if failed_relations:
        print(f"\nWarning: Failed to define {len(failed_relations)} relations")
    
    # Phase 3: Canonicalize schema using parallel implementation
    canonical_triples, canonical_map = parallel_canonicalize_schema(
        normalized_triples,
        relation_definitions,
        batch_size=10,
        max_workers=4
    )
    
    # Save results to files
    if canonical_triples:
        # Print canonical triples before saving
        print("\n=== Final Canonical Triples to be saved ===")
        print(f"Total number of canonical triples: {len(canonical_triples)}")
        print("Sample of first 5 triples:")
        for i, triple in enumerate(canonical_triples[:5]):
            print(f"{i+1}. Subject: {triple['subject']}")
            print(f"   Predicate: {triple['predicate']}")
            print(f"   Object: {triple['object']}\n")
        
        # Save canonical triples
        with open('canonical_triples.json', 'w') as f:
            json.dump(canonical_triples, f, indent=2)
        print("Saved canonical triples to canonical_triples.json")
        
        # Save relation definitions and mapping
        schema_info = {
            'relation_definitions': relation_definitions,
            'canonical_mapping': canonical_map,
            'failed_relations': failed_relations
        }
        with open('schema_info.json', 'w') as f:
            json.dump(schema_info, f, indent=2)
        print("Saved schema information to schema_info.json")
        
        # Save normalized triples for comparison
        with open('normalized_triples.json', 'w') as f:
            json.dump(normalized_triples, f, indent=2)
        print("Saved normalized triples to normalized_triples.json")

        # Build and visualize the knowledge graph
        print("\nBuilding and visualizing knowledge graph...")
        G, qa_chain, graphml_path = build_and_visualize_knowledge_graph(company_name, "canonical_triples.json", "graph_output")
        print("Knowledge graph processing complete!")

        # Example usage of graph QA
        if qa_chain:
            print("\nTesting Graph QA capabilities...")
            sample_queries = [
                "What are the main revenue sources for the company?",
                "What are the key risks mentioned in the document?",
                "What is the company's business strategy?",
            ]
            
            for query in sample_queries:
                answer = query_knowledge_graph(qa_chain, query)
                print(f"\nQ: {query}")
                print(f"A: {answer}")
        else:
            print("Graph QA system is not available. Skipping QA testing.")

if __name__ == "__main__":
    main()

