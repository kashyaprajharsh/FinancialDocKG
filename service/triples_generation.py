from Models.llm import generate_response
from schemas.models import TripleList
from utils.prompts import extraction_system_prompt, extraction_user_prompt_template
from typing import List, Tuple
import time
import re
import math
import concurrent.futures
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' data...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK 'wordnet' data...")
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK 'averaged_perceptron_tagger' data...")
    nltk.download('averaged_perceptron_tagger', quiet=True)

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
        
        print("2. Sending request to LLM...")
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = generate_response(system_prompt, TripleList)
                
                extracted_triples = []
                for triple in response.triples:
                    triple_dict = triple.dict()
                    triple_dict['chunk'] = chunk_idx  
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
                    retry_delay *= 2  
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
    Normalize (including lemmatization) and deduplicate knowledge graph triples.
    
    Args:
        all_extracted_triples: List of extracted triples
        
    Returns:
        List of normalized and deduplicated triples
    """
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        """Map Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_text(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        tokens = word_tokenize(text.lower())
        tagged_tokens = nltk.pos_tag(tokens)

        lemmatized_tokens = []
        for word, tag in tagged_tokens:
            wn_tag = get_wordnet_pos(tag)
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wn_tag))

        # Simple lemmatization (without POS tagging for speed)
        # lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    normalized_triples = []
    seen_triples = set()  
    original_count = len(all_extracted_triples)
    empty_removed_count = 0
    duplicates_removed_count = 0

    print(f"Starting normalization (with lemmatization) and de-duplication of {original_count} triples...")
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
            # 1. Basic Normalize (strip, lower, predicate space)
            sub_clean = subject_raw.strip()
            pred_clean = re.sub(r'\s+', ' ', predicate_raw.strip()).strip()
            obj_clean = object_raw.strip()
            if show_example:
                print(f"Cleaned: SUB='{sub_clean}', PRED='{pred_clean}', OBJ='{obj_clean}'")

            # 2. Lemmatize
            normalized_sub = lemmatize_text(sub_clean)
            normalized_pred = lemmatize_text(pred_clean)
            normalized_obj = lemmatize_text(obj_clean)

            if show_example:
                print(f"Lemmatized: SUB='{normalized_sub}', PRED='{normalized_pred}', OBJ='{normalized_obj}'")

            # 3. Filter Empty
            if normalized_sub and normalized_pred and normalized_obj:
                triple_identifier = (normalized_sub, normalized_pred, normalized_obj)
                
                # 4. De-duplicate
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
                        print("Status: Kept (New Unique Lemmatized Triple)")
                else:
                    duplicates_removed_count += 1
                    if show_example:
                        print("Status: Discarded (Duplicate after lemmatization)")
            else:
                empty_removed_count += 1
                if show_example:
                    print("Status: Discarded (Empty component after lemmatization)")
        else:
            empty_removed_count += 1  # Count non-string/missing as needing removal
            if show_example:
                print("Status: Discarded (Non-string or missing component)")
        processed_count += 1

    print(f"\n... Finished processing {processed_count} triples.")
    print(f"\nNormalization Summary (with Lemmatization):")
    print(f"Original triples: {original_count}")
    print(f"Empty/invalid triples removed: {empty_removed_count}")
    print(f"Duplicate triples removed (post-lemmatization): {duplicates_removed_count}")
    print(f"Final unique triples: {len(normalized_triples)}")
    
    return normalized_triples


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
            triples, failures = process_chunk(
                chunk_idx,
                chunk,  
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
                'chunk': chunk[:100] + '...'  
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