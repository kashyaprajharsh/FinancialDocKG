# Detailed Project Documentation: Automated Knowledge Graph Builder

This document provides an in-depth walkthrough of the entire process involved in the Automated Knowledge Graph Builder project, detailing each phase, the techniques used, the rationale behind them, and their advantages, illustrated with examples.

## Phase 1: Document Ingestion & Preparation

### Step 1.1: PDF Loading and Initial Text Extraction

*   **What:** Loads the input PDF document provided by the user.
*   **How:** Uses the `pdfplumber` library to open and read the PDF file page by page (`pdfplumber.open(pdf_path)`).
*   **Why:** This is the necessary first step to access the document's content. `pdfplumber` is chosen because it's effective at extracting text based on character positioning and also provides robust tools for detecting and extracting structured data like tables, which is crucial for this project.
*   **Advantages:** Compared to simpler text extraction methods, `pdfplumber` offers better handling of complex layouts and explicit table detection, reducing the risk of losing information or extracting jumbled text from tables.
*   **Example:** A user uploads `CompanyReport.pdf`. The system opens this file using `pdfplumber`.

### Step 1.2: Table Identification and Extraction

*   **What:** Identifies and extracts tabular data separately from the regular text flow on each page.
*   **How:** Within each page loop, `page.extract_tables()` from `pdfplumber` is called. This function returns a list of tables found on the page, where each table is represented as a list of lists (rows and cells).
*   **Why:** Tables contain highly structured, dense information that is often critical (e.g., financial figures, lists of subsidiaries). Treating them separately prevents their content from being merged nonsensically with surrounding paragraphs during simple text extraction and allows for specific processing later.
*   **Advantages:** Preserves the inherent row/column structure of tables, ensuring that relationships between data points within a table are maintained. Allows the system to know *which* text came from a table.
*   **Example:** On page 5 of `CompanyReport.pdf`, a table listing quarterly revenue is detected. `page.extract_tables()` returns `[[['Region', 'Q1 Revenue', 'Q2 Revenue'], ['North', '1.2M', '1.4M'], ['South', '0.8M', '0.9M']]]`.

### Step 1.3: Textual Conversion of Tables and Marking

*   **What:** Converts the extracted table data (list of lists) into a plain text format and embeds special markers around it within the overall extracted text.
*   **How:** Iterates through the rows and cells of each extracted table. Cells are joined into rows (e.g., using tabs `\t` as separators), and rows are joined by newlines. Special marker lines like `--- TABLE P-I ---` and `--- END TABLE P-I ---` (where P is page number, I is table index on that page) are added before and after the formatted table text.
*   **Why:** LLMs process plain text. This step transforms the structured table into a text format the LLM can read while the markers clearly delineate the table's boundaries. This allows downstream processes (like chunking) to be aware of table locations.
*   **Advantages:** Makes table data digestible for LLMs. The markers are essential for the custom chunking logic (Step 1.4) to avoid splitting tables inappropriately.
*   **Example:** The revenue table from Step 1.2 is converted to the text:
    ```
    --- TABLE 5-1 ---
    Region	Q1 Revenue	Q2 Revenue
    North	1.2M	1.4M
    South	0.8M	0.9M
    --- END TABLE 5-1 ---
    ```
    This text block is inserted into the main text extracted from page 5.

### Step 1.4: Full Text Aggregation and Basic Cleaning

*   **What:** Combines the regular text extracted from each page with the specially formatted table text blocks. Performs minor cleaning.
*   **How:** Concatenates the `page.extract_text()` output and the formatted table text for every page into a single large string. A simple regex (`re.sub(r'\n{3,}', '\n\n', text).strip()`) is used to collapse sequences of three or more newlines into double newlines, preserving paragraph breaks but removing excessive vertical whitespace.
*   **Why:** Creates the complete textual representation of the document that will be fed into the next phase (chunking and extraction). Basic cleaning improves readability and slightly reduces noise.
*   **Advantages:** Ensures all content (text and tables) is available for processing in a single stream. Cleaning prevents potential issues with overly sparse text.
*   **Example:** The text and marked table blocks from all pages of `CompanyReport.pdf` are combined into one long string `full_document_text`.

## Phase 2: Text Chunking for LLM Processing

### Step 2.1: Table-Aware Recursive Text Splitting

*   **What:** Divides the `full_document_text` into smaller, overlapping chunks suitable for processing by an LLM, while attempting to keep table structures intact.
*   **How:** Uses `langchain_text_splitters.RecursiveCharacterTextSplitter`. Key configurations:
    *   `chunk_size`: Target size for each chunk (e.g., 1536 characters). LLMs have context window limits, so text must be split.
    *   `chunk_overlap`: Amount of text repeated at the end of one chunk and the beginning of the next (e.g., 20% of chunk size). This helps maintain context across chunk boundaries.
    *   `separators`: A prioritized list (`["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]`). The splitter tries to split on the first separator in the list that allows it to create a valid chunk. This prioritizes splitting between paragraphs, then lines, then sentences, etc.
    *   **Table Handling Logic:**
        1.  Identifies large tables (e.g., >80% of `chunk_size`) using the `--- TABLE ... ---` markers.
        2.  Temporarily replaces these large tables with placeholders (`[TABLE_PLACEHOLDER_i]`) in the main text.
        3.  Splits the main text (with placeholders) using the configured `RecursiveCharacterTextSplitter`.
        4.  Chunks the *content* of the large tables separately, often using a *larger* chunk size (e.g., `chunk_size * 3`) to preserve more internal table structure.
        5.  Replaces placeholders in the main text chunks with either the original table content (if small enough) or a note indicating it was processed separately.
        6.  Appends the separately processed large-table chunks to the final list of chunks.
*   **Why:** LLMs perform better on smaller, focused text segments. Simple splitting can break sentences, paragraphs, or crucial structures like tables mid-way, destroying context. Recursive splitting with semantic separators (like paragraphs) is better. The custom table handling is critical because tables often exceed typical chunk sizes but need to be analyzed coherently.
*   **Advantages:** Produces manageable text segments for the LLM. Overlap preserves local context. Table-aware logic significantly increases the chance that table data is processed meaningfully, either within a regular chunk or as dedicated table chunks, preventing data loss or nonsensical splitting.
*   **Example:** A section of text containing a small table might become one chunk. A section containing a very large table might result in one chunk with the surrounding text and a placeholder `[Table was too large and is processed separately]`, plus one or more additional chunks containing only the (larger-chunked) content of that table.

## Phase 3: Knowledge Triple Extraction

### Step 3.1: Parallel Chunk Processing Setup

*   **What:** Prepares to process the text chunks in parallel to speed up the computationally intensive LLM calls.
*   **How:** Uses Python's `concurrent.futures` library (likely `ThreadPoolExecutor` or `ProcessPoolExecutor` based on `max_workers` argument). The list of text chunks generated in Phase 2 is divided into smaller batches (e.g., `batch_size=10`). Each batch is submitted as a separate task to the executor pool.
*   **Why:** Calling the LLM for each chunk sequentially would be very slow, especially for large documents with many chunks. Parallel processing significantly reduces the overall runtime by making multiple LLM calls concurrently.
*   **Advantages:** Drastically improves the throughput and speed of the triple extraction phase, making the system practical for larger documents.
*   **Example:** If there are 100 chunks and `max_workers=4`, the system might process 4 chunks (or 4 batches of chunks) simultaneously instead of one after another.

### Step 3.2: LLM Prompting for Triples

*   **What:** For each individual chunk, sends a request to a Large Language Model (LLM) asking it to extract knowledge triples (Subject, Predicate, Object).
*   **How:**
    1.  A **system prompt** is constructed. This prompt instructs the LLM on its task: identify factual statements, extract them as triples, focus on relevance to the specified `company_name`, and format the output according to a predefined schema (likely JSON). The actual `chunk_text` is embedded within this prompt (e.g., using f-strings or a template like `extraction_system_prompt.format(text_chunk=chunk_text, company_name=company_name)`).
    2.  The `Models.llm.generate_response` function is called, passing the formatted system prompt and specifying the expected output structure (e.g., `TripleList`, a Pydantic model defining a list of triples with 'subject', 'predicate', 'object' fields).
    3.  This function handles the actual API call to the LLM service (e.g., Google Vertex AI Gemini API).
*   **Why:** LLMs excel at understanding natural language and identifying semantic relationships within text. Prompting them is the core mechanism for converting unstructured text into structured triples. Specifying the output schema (like `TripleList`) helps ensure the LLM returns data in a usable format.
*   **Advantages:** Leverages the power of LLMs to understand context and extract diverse, implicit relationships that rule-based systems would miss. Relatively adaptable to different domains through prompt engineering.
*   **Example:** For a chunk containing "Acme Corp acquired Beta Inc in 2023.", the LLM (guided by the prompt) should ideally return a triple like `{"subject": "Acme Corp", "predicate": "acquired", "object": "Beta Inc"}` (potentially others like subject: Beta Inc, predicate: was acquired by, object: Acme Corp; or subject: Acme Corp acquisition of Beta Inc, predicate: occurred in, object: 2023).

### Step 3.3: Response Parsing and Retry Logic

*   **What:** Parses the LLM's response, handles potential errors, and retries failed requests.
*   **How:**
    1.  The `generate_response` function attempts to parse the LLM's raw output into the specified Pydantic model (`TripleList`).
    2.  If parsing is successful, the list of triple objects is converted to a list of dictionaries. The source `chunk_idx` is added to each dictionary.
    3.  If the API call fails (e.g., network error, rate limit, server error) or parsing fails, the `process_chunk` function implements a retry loop (e.g., up to `max_retries=3` attempts) with exponential backoff (increasing delay between retries, e.g., `retry_delay *= 2`).
    4.  If all retries fail, the chunk is marked as a failure, recording the error message and the problematic `chunk_text`.
*   **Why:** LLM APIs can occasionally fail or return malformed responses. Robust error handling and retry logic are essential for reliability. Adding the `chunk_idx` links the extracted triples back to their source text, which can be useful for context or debugging.
*   **Advantages:** Increases the success rate of triple extraction by automatically recovering from transient errors. Tracks failures systematically.
*   **Example:** If the first API call for Chunk 5 times out, the system waits 2 seconds, tries again. If that fails, it waits 4 seconds, tries a third time. If that also fails, Chunk 5 is added to the `failures` list.

### Step 3.4: Aggregation of Results

*   **What:** Collects all successfully extracted triples and all recorded failures from the parallel processing of chunks.
*   **How:** The main `parallel_process_chunks` function waits for all submitted batch tasks to complete and aggregates the lists of triples and failures returned by each task into two overall lists: `all_extracted_triples` and `all_failures`.
*   **Why:** Consolidates the results from the parallel execution into a single dataset for the next phase.
*   **Advantages:** Simple aggregation step ensuring all results are captured. Results are cached to disk for performance optimization in subsequent runs.
*   **Example:** The lists of triples extracted from Chunk 1, Chunk 2, etc., are combined into one large list. Any failures recorded for Chunk 5, Chunk 12, etc., are combined into a separate list.

**Note on AI Models Used:** This project uses Google's Gemini API for all LLM tasks (triple extraction, relation definitions) and BGE (BAAI General Embedding) models for generating embeddings used in canonicalization steps. These specific model choices offer a balance of performance and cost-effectiveness.

## Phase 4: Normalization and Deduplication

### Step 4.1: Lemmatization Setup

*   **What:** Initializes tools for lemmatization, the process of reducing words to their base or dictionary form.
*   **How:** Uses the `nltk` (Natural Language Toolkit) library. Specifically:
    1.  Initializes `WordNetLemmatizer()`.
    2.  Downloads necessary `nltk` data files (`punkt` for tokenization, `wordnet` for the lemmatization dictionary, `averaged_perceptron_tagger` for Part-of-Speech tagging) if they are not already present.
    3.  Defines a helper function (`get_wordnet_pos`) to map the Part-of-Speech (POS) tags generated by `nltk.pos_tag` (like 'NN' for noun, 'VB' for verb) to the format expected by WordNet (e.g., `wordnet.NOUN`, `wordnet.VERB`).
*   **Why:** Words can appear in different forms (e.g., "run", "runs", "ran", "running"). To treat these as the same concept during deduplication and canonicalization, we need to reduce them to a common base form ("run"). Lemmatization is generally preferred over stemming as it produces actual dictionary words. Using POS tags improves lemmatization accuracy (e.g., lemmatizing "meeting" as a noun vs. a verb).
*   **Advantages:** Leads to more effective deduplication and better grouping of semantically related concepts in later canonicalization phases. Using POS tags makes the lemmatization more linguistically accurate.
*   **Example:** `nltk` data is downloaded. The lemmatizer is ready to convert "acquired" to "acquire" (verb) and "companies" to "company" (noun).

### Step 4.2: Triple Normalization (Cleaning & Lemmatization)

*   **What:** Cleans and lemmatizes the subject, predicate, and object text within each extracted triple.
*   **How:** Iterates through each triple in `all_extracted_triples`:
    1.  **Basic Cleaning:** Removes leading/trailing whitespace (`strip()`) from subject, predicate, and object. Collapses multiple spaces within the predicate (`re.sub(r'\s+', ' ', ...)`).
    2.  **Lemmatization:** For each component (subject, predicate, object):
        *   Tokenizes the text (splits into words) using `nltk.word_tokenize`.
        *   Performs POS tagging on the tokens using `nltk.pos_tag`.
        *   Lemmatizes each token using `lemmatizer.lemmatize(word, pos=wn_tag)`, providing the WordNet-compatible POS tag obtained via `get_wordnet_pos`.
        *   Joins the lemmatized tokens back into a string.
*   **Why:** Applies the lemmatization prepared in the previous step. Basic cleaning ensures consistency before lemmatization. This step standardizes the textual representation of the triple components.
*   **Advantages:** Creates a canonical textual form for each component, crucial for accurate deduplication.
*   **Example:**
    *   Original: `{"subject": "  Acme Corp Incorporated ", "predicate": "acquired the assets of", "object": " Beta Companies "}`
    *   Cleaned: `{"subject": "Acme Corp Incorporated", "predicate": "acquired the assets of", "object": "Beta Companies"}`
    *   Lemmatized: `{"subject": "acme corp incorporated", "predicate": "acquire the asset of", "object": "beta company"}` (simplified example, actual POS tagging/lemmatization might differ slightly).

### Step 4.3: Filtering Empty Triples

*   **What:** Removes triples where any component (subject, predicate, or object) becomes empty after normalization.
*   **How:** Checks if the normalized subject, predicate, *and* object strings are all non-empty. If any are empty, the triple is discarded.
*   **Why:** An empty component makes a triple meaningless. This can happen if the original component was just whitespace or symbols that get removed during cleaning/lemmatization.
*   **Advantages:** Ensures only valid, complete triples proceed to the next steps.
*   **Example:** If a triple somehow ended up as `{"subject": "Acme Corp", "predicate": "", "object": "Location X"}` after normalization, it would be filtered out.

### Step 4.4: Deduplication

*   **What:** Removes duplicate triples based on their normalized and lemmatized components.
*   **How:** Maintains a `set` called `seen_triples`. For each normalized triple, it creates a tuple `(normalized_sub, normalized_pred, normalized_obj)`. If this tuple is *not* already in `seen_triples`, the triple is added to the final `normalized_triples` list, and the tuple is added to `seen_triples`. If the tuple *is* already present, the current triple is discarded as a duplicate.
*   **Why:** The LLM might extract the same fact multiple times from different parts of the text or express the same fact in slightly different ways that normalize to the same form. This step ensures each unique fact (in its normalized form) is represented only once.
*   **Advantages:** Reduces redundancy in the knowledge graph, making it cleaner and more efficient. Improves the accuracy of counts and analyses performed on the graph.
*   **Example:** If the LLM extracted both `(Acme Corp, acquired, Beta Inc)` and `(Acme Corp, acquisition of, Beta Inc)` and both normalize/lemmatize to `('acme corp', 'acquire', 'beta inc')`, only the first one encountered would be kept.

## Phase 5: Schema Definition and Canonicalization

This phase aims to consolidate different phrasings of the same relationship (predicate) into a single, canonical form.

### Step 5.1: Identify Unique Predicates

*   **What:** Extracts a list of all unique predicate strings present in the normalized and deduplicated triples.
*   **How:** Iterates through the `normalized_triples` list and adds each triple's `'predicate'` value to a Python `set`. The final set contains each distinct predicate string exactly once.
*   **Why:** We need to understand the vocabulary of relationships extracted by the LLM before we can attempt to standardize it.
*   **Advantages:** Provides a clear inventory of all the relationship types identified in the document.
*   **Example:** If triples include predicates like `"acquire"`, `"purchase"`, `"have location"`, `"located in"`, `"ceo of"`, `"acquire"`, the unique set would be `{"acquire", "purchase", "have location", "located in", "ceo of"}`.

### Step 5.2: Generate Relation Definitions (LLM)

*   **What:** For each unique predicate, uses an LLM to generate a concise definition and assign a category.
*   **How:**
    1.  Iterates through the unique predicates (potentially in parallel batches using `concurrent.futures` via `parallel_schema_processing`).
    2.  For each predicate, constructs a specific prompt asking the LLM to: a) provide a one-sentence definition suitable for grouping similar relations in a business/financial context, and b) assign a category from a predefined list (e.g., `[LOCATION, OWNERSHIP, ROLE, FINANCIAL_METRIC, ...]`).
    3.  Calls the `Models.llm.generate_response` function with this prompt, expecting a string output formatted like `"Category: <CATEGORY_NAME> | Definition: <Generated definition>"`.
    4.  Parses the response to extract the definition and category. Stores the *definition* associated with the original predicate string (e.g., in a dictionary `relation_definitions`).
*   **Why:** The raw predicate strings extracted earlier (like `"acquire"`, `"purchase"`) might be synonyms or closely related. Simply comparing these strings is insufficient. By generating a *definition* for each, we capture its semantic meaning, which can then be compared more effectively using embeddings.
*   **Advantages:** Moves beyond surface-level string matching to semantic understanding. The generated definitions provide a richer basis for identifying truly similar relationships. Categorization can provide additional metadata (though it doesn't seem directly used for canonicalization in the current code).
*   **Example:**
    *   For predicate `"acquire"`, LLM might return: `"Category: OWNERSHIP | Definition: To gain possession or control of an asset or company."`
    *   For predicate `"purchase"`, LLM might return: `"Category: OWNERSHIP | Definition: To obtain something by paying money for it."`
    *   For predicate `"located in"`, LLM might return: `"Category: LOCATION | Definition: Specifies the geographical place where an entity resides or operates."`

### Step 5.3: Generate Embeddings for Definitions

*   **What:** Converts the textual definitions generated in the previous step into numerical vector representations (embeddings).
*   **How:** Uses the `Models.embed.get_embeddings_batch` function. This function takes the list of definition strings and calls an embedding model API (likely via Google Cloud AI Platform/Vertex AI) to get a vector embedding for each definition.
*   **Why:** Embeddings capture the semantic meaning of text in a numerical format. We need these vectors to mathematically calculate the similarity between different relation definitions.
*   **Advantages:** Allows quantitative comparison of semantic meaning. State-of-the-art embedding models can capture subtle nuances in meaning.
*   **Example:** The definition `"To gain possession or control of an asset or company."` is converted into a high-dimensional vector like `[0.12, -0.34, 0.56, ..., -0.01]`. The definition `"To obtain something by paying money for it."` is converted into a similar vector `[0.15, -0.30, 0.51, ..., 0.05]`.

### Step 5.4: Compute Similarity Between Definitions

*   **What:** Calculates the similarity between the vector embeddings of every pair of relation definitions.
*   **How:** Iterates through all pairs of embeddings generated in Step 5.3. For each pair, uses `Models.embed.compute_similarity` (which likely calculates the cosine similarity between the two vectors). Cosine similarity measures the angle between two vectors, with values closer to 1 indicating higher similarity.
*   **Why:** This step quantifies how semantically close the meanings of the original predicates are, based on their LLM-generated definitions.
*   **Advantages:** Provides a numerical score representing semantic relatedness, allowing for threshold-based grouping.
*   **Example:**
    *   `similarity(embedding("acquire" definition), embedding("purchase" definition))` might yield `0.92`.
    *   `similarity(embedding("acquire" definition), embedding("located in" definition))` might yield `0.15`.

### Step 5.5: Group Similar Relations

*   **What:** Groups predicates whose definitions are semantically similar above a defined threshold.
*   **How:**
    1.  Sets a `similarity_threshold` (e.g., 0.90).
    2.  Iterates through each relation and its definition embedding.
    3.  Compares its embedding similarity (calculated in Step 5.4) to all other relations not yet processed.
    4.  If a pair's similarity exceeds the `similarity_threshold`, they are considered part of the same group.
    5.  A simple clustering approach is used: the first relation encountered in a group becomes the representative (canonical) name for that group. All other relations found to be similar above the threshold are mapped to this canonical name.
*   **Why:** This is the core canonicalization step. It uses the semantic similarity scores to identify and group different predicate strings that represent the same underlying relationship concept.
*   **Advantages:** Consolidates synonymous or near-synonymous predicates, creating a cleaner, more consistent graph schema. Robust to variations in phrasing used by the initial LLM extraction.
*   **Example:** With a threshold of 0.90:
    *   `"acquire"` (similarity to self is 1.0) is processed first. It's compared to `"purchase"`. Similarity is 0.92 (> 0.90). So, `"purchase"` is grouped with `"acquire"`.
    *   It's compared to `"located in"`. Similarity is 0.15 (< 0.90). No grouping.
    *   It's compared to `"have location"`. Assume similarity is 0.18 (< 0.90). No grouping.
    *   It's compared to `"ceo of"`. Assume similarity is 0.05 (< 0.90). No grouping.
    *   Next, `"located in"` is processed (assume it wasn't already grouped). It's compared to `"have location"`. Assume similarity is 0.95 (> 0.90). `"have location"` is grouped with `"located in"`.
    *   This results in groups: `{"acquire", "purchase"}` and `{"located in", "have location"}` and `{"ceo of"}`.

### Step 5.6: Assign Canonical Predicate Names

*   **What:** Creates a mapping from each original predicate to its chosen canonical predicate name.
*   **How:** Based on the grouping in Step 5.5. If a relation was grouped, it maps to the representative name chosen for its group (e.g., the first one encountered). If a relation wasn't similar enough to any other, it maps to itself.
*   **Why:** Formalizes the result of the grouping process into a lookup table needed to update the triples.
*   **Advantages:** Provides a clear, usable mapping for the final update step.
*   **Example:** The mapping `canonical_map` would be:
    *   `"acquire"` -> `"acquire"`
    *   `"purchase"` -> `"acquire"`
    *   `"located in"` -> `"located in"`
    *   `"have location"` -> `"located in"`
    *   `"ceo of"` -> `"ceo of"`

### Step 5.7: Update Triples with Canonical Predicates

*   **What:** Replaces the original predicate string in each triple with its corresponding canonical predicate name.
*   **How:** Iterates through the `normalized_triples` list. For each triple, it looks up the triple's original `'predicate'` in the `canonical_map` created in Step 5.6 and updates the `'predicate'` value with the canonical name found in the map.
*   **Why:** Applies the canonicalization results to the actual data, ensuring all triples representing the same relationship type now use the exact same predicate string.
*   **Advantages:** Creates the final set of triples with a standardized, consistent schema.
*   **Example:**
    *   A triple `{"subject": ..., "predicate": "purchase", "object": ...}` becomes `{"subject": ..., "predicate": "acquire", "object": ...}`.
    *   A triple `{"subject": ..., "predicate": "have location", "object": ...}` becomes `{"subject": ..., "predicate": "located in", "object": ...}`.
    *   A triple `{"subject": ..., "predicate": "acquire", "object": ...}` remains unchanged.

## Phase 6: Entity Canonicalization

This phase mirrors Phase 5 but operates on the subjects and objects (entities) instead of the predicates (relations).

### Step 6.1: Identify Unique Entities

*   **What:** Extracts a list of all unique entity strings (both subjects and objects) present in the triples (which now have canonical predicates).
*   **How:** Iterates through the `canonical_triples` list from Phase 5. Adds both the `'subject'` value and the `'object'` value of each triple to a Python `set`.
*   **Why:** To get an inventory of all distinct entity mentions before attempting to merge variations of the same real-world entity.
*   **Advantages:** Provides the complete list of unique entity strings found in the document.
*   **Example:** The set might contain `{"acme corp", "acme corporation", "beta inc.", "beta incorporated", "new york", "ny", "mr. john doe", "2023", "$1.5 million"}`.

### Step 6.2: Filter Entities for Canonicalization

*   **What:** Filters the unique entities to exclude those containing numerical digits, which are typically not subject to semantic canonicalization.
*   **How:** Uses regular expressions (`re.search(r'\d', normalized_entity)`) to identify entities containing any digits after normalization.
*   **Why:** Entities containing digits (like dates, financial figures, percentages, IDs) typically don't have meaningful semantic synonyms in the way that company names or people's names do. Including them in semantic similarity comparisons can lead to incorrect merging.
*   **Advantages:** Improves canonicalization accuracy and efficiency by focusing only on named entities likely to have variations. Prevents nonsensical merging like `"2022"` with `"2023"` or `"$1.5 million"` with `"$1.6 million"`.
*   **Example:** `"acme corp"`, `"acme corporation"`, `"new york"`, `"ny"` would be included for semantic canonicalization. `"2023"`, `"$1.5 million"`, `"10%"` would be skipped (mapped to themselves in the final canonical map).

### Step 6.3: Generate Embeddings for Filtered Entities

*   **What:** Converts the filtered unique entity strings (without digits) into numerical vector representations (embeddings) using the BGE embedding model.
*   **How:** Uses `Models.embed.get_embeddings_batch`, passing the list of filtered unique entity strings (those without digits). Similar to Step 5.3.
*   **Why:** As with relation definitions (Phase 5), embeddings capture the semantic meaning of entity names numerically, enabling quantitative similarity comparisons.
*   **Advantages:** Allows quantitative comparison of potentially different names referring to the same entity. Focusing only on filtered entities improves processing efficiency.
*   **Example:** `"acme corp"` becomes `[0.6, 0.1, -0.2, ...]`, `"acme corporation"` becomes `[0.58, 0.12, -0.21, ...]`.

### Step 6.4: Compute Similarity Between Entities

*   **What:** Calculates the semantic similarity between the vector embeddings of every pair of unique entity strings.
*   **How:** Uses `Models.embed.compute_similarity` (likely cosine similarity) on all pairs of entity embeddings. Similar to Step 5.4.
*   **Why:** Quantifies how semantically close different entity mentions are.
*   **Advantages:** Provides a numerical score for identifying potential co-referent entities.
*   **Example:**
    *   `similarity(embedding("acme corp"), embedding("acme corporation"))` might yield `0.98`.
    *   `similarity(embedding("new york"), embedding("ny"))` might yield `0.95`.
    *   `similarity(embedding("acme corp"), embedding("beta inc."))` might yield `0.05`.

### Step 6.5: Group Similar Entities

*   **What:** Groups entity strings whose embeddings are similar above a defined threshold.
*   **How:** Sets a `similarity_threshold` (e.g., 0.85 - might be different from the schema threshold). Uses a similar clustering approach as Step 5.5: iterates through entities, compares similarity scores, and groups those above the threshold. The first entity encountered in a group often becomes the representative canonical name.
*   **Why:** To identify and group different textual mentions (e.g., `"Acme Corp"`, `"Acme Corporation"`, `"Acme"`) that refer to the same underlying real-world entity.
*   **Advantages:** Resolves entity variations, leading to a cleaner knowledge graph where each node represents a unique entity. Crucial for accurate network analysis and query results.
*   **Example:** With threshold 0.85:
    *   `"acme corp"` and `"acme corporation"` (similarity 0.98) are grouped.
    *   `"beta inc."` and `"beta incorporated"` (assume similarity 0.97) are grouped.
    *   `"new york"` and `"ny"` (similarity 0.95) are grouped.
    *   `"mr. john doe"` remains ungrouped.

### Step 6.6: Assign Canonical Entity Names

*   **What:** Creates a mapping from each original entity string to its chosen canonical entity name.
*   **How:** Based on the grouping in Step 6.5. Entities in a group map to the group's representative name. Ungrouped entities map to themselves.
*   **Why:** Formalizes the entity grouping into a lookup table.
*   **Advantages:** Provides the necessary mapping to update the triples.
*   **Example:** The `canonical_entity_map` might be:
    *   `"acme corp"` -> `"acme corp"`
    *   `"acme corporation"` -> `"acme corp"`
    *   `"beta inc."` -> `"beta inc."`
    *   `"beta incorporated"` -> `"beta inc."`
    *   `"new york"` -> `"new york"`
    *   `"ny"` -> `"new york"`
    *   `"mr. john doe"` -> `"mr. john doe"`

### Step 6.7: Update Triples with Canonical Entities

*   **What:** Replaces the original subject and object strings in each triple with their corresponding canonical entity names.
*   **How:** Iterates through the list of triples (which already have canonical predicates from Phase 5). For each triple, it looks up the triple's original `'subject'` in the `canonical_entity_map` and updates it. It then does the same for the triple's original `'object'`.
*   **Why:** Applies the entity canonicalization results, ensuring that all mentions of the same entity now use the exact same string identifier in the subject and object fields.
*   **Advantages:** Creates the final set of triples where both relations and entities are standardized and consistent.
*   **Example:**
    *   Triple `{"subject": "acme corporation", "predicate": "acquire", "object": "beta incorporated"}` becomes `{"subject": "acme corp", "predicate": "acquire", "object": "beta inc."}`.
    *   Triple `{"subject": "beta inc.", "predicate": "located in", "object": "ny"}` becomes `{"subject": "beta inc.", "predicate": "located in", "object": "new york"}`.

## Phase 7: Knowledge Graph Construction

This phase takes the fully cleaned, normalized, and canonicalized triples and builds a formal graph structure.

### Step 7.1: Initialize Graph Object

*   **What:** Creates an empty graph object using a graph library.
*   **How:** Uses the `networkx` library, typically by instantiating a `nx.DiGraph()` for a directed graph, as relationships usually have a direction (e.g., A acquires B is different from B acquires A).
*   **Why:** `networkx` is a standard and powerful Python library for graph creation, manipulation, and analysis. A directed graph (`DiGraph`) is appropriate for representing subject-predicate-object triples where the predicate implies directionality from subject to object.
*   **Advantages:** Provides a robust data structure with numerous built-in algorithms for graph analysis (centrality, paths, communities, etc.) if needed.
*   **Example:** `G = nx.DiGraph()` creates an empty graph container `G`.

### Step 7.2: Add Nodes and Edges from Canonical Triples

*   **What:** Populates the graph object with nodes (entities) and edges (relationships) based on the final list of canonical triples.
*   **How:** Iterates through the `canonical_triples` list generated at the end of Phase 6. For each triple `{"subject": s, "predicate": p, "object": o}`:
    1.  Adds the subject `s` as a node: `G.add_node(s)` (if not already present).
    2.  Adds the object `o` as a node: `G.add_node(o)` (if not already present).
    3.  Adds a directed edge from node `s` to node `o`: `G.add_edge(s, o, label=p)`. The predicate `p` is typically stored as an attribute (e.g., `label`) of the edge.
*   **Why:** This directly translates the structured triple information into the graph format. Each unique entity becomes a node, and each fact becomes a labeled edge connecting the relevant entity nodes.
*   **Advantages:** Creates a concrete graph representation of the extracted knowledge, suitable for visualization and computational analysis.
*   **Example:** For the triple `{"subject": "acme corp", "predicate": "acquire", "object": "beta inc."}`:
    *   Node `"acme corp"` is added/ensured.
    *   Node `"beta inc."` is added/ensured.
    *   A directed edge is added from `"acme corp"` to `"beta inc."`, with the attribute `label="acquire"`.

## Phase 8: Visualization and Storage

This phase makes the results accessible and persistent.

### Step 8.1: Build Interactive HTML Visualization

*   **What:** Creates an interactive HTML file that visualizes the constructed knowledge graph.
*   **How:** Uses the `pyvis` library. 
    1.  Creates a `pyvis.network.Network` object, potentially configuring height, width, physics layout options (like `forceAtlas2Based` for layout), and interaction options (navigation buttons, keyboard controls).
    2.  Iterates through the nodes in the `networkx` graph `G` (`G.nodes()`) and adds them to the `pyvis` network (`net.add_node(...)`).
    3.  Iterates through the edges in the `networkx` graph `G` (`G.edges(data=True)`) and adds them to the `pyvis` network (`net.add_edge(...)`), using the edge label (predicate) as the edge title/label.
    4.  Saves the `pyvis` network to an HTML file (`net.save_graph("knowledge_graph_interactive.html")`).
*   **Why:** A visual representation makes the complex network of relationships much easier for humans to explore and understand than raw triples or graph data files. `pyvis` creates interactive visualizations suitable for web browsers.
*   **Advantages:** Provides an intuitive way for stakeholders to explore the extracted knowledge, zoom, pan, and inspect nodes/edges. The physics-based layout helps reveal clusters and important nodes.
*   **Example:** An HTML file is generated that, when opened in a browser, displays nodes like "acme corp", "beta inc.", "new york" and edges like "acquire", "located in" connecting them. Users can drag nodes and hover to see details.

### Step 8.2: Save GraphML Data

*   **What:** Saves the constructed graph structure in a standard, machine-readable format.
*   **How:** Uses `networkx`'s built-in function to save the graph `G` to a GraphML file: `nx.write_graphml(G, "knowledge_graph.graphml")`.
*   **Why:** GraphML is a standard XML-based format for storing graph structures. Saving in this format allows the graph to be loaded later by `networkx` itself, or imported into other graph analysis tools (like Gephi) or databases that support it.
*   **Advantages:** Provides a persistent, standardized representation of the graph for reuse, sharing, or further analysis with external tools.
*   **Example:** A `knowledge_graph.graphml` file is created containing XML tags defining all the nodes and edges with their attributes.

### Step 8.3: Save Supporting Data (Canonical Triples, Schema Info, Chunks)

*   **What:** Saves the intermediate and final data artifacts generated during the process.
*   **How:** Uses standard Python file I/O and the `json` library:
    *   Saves the final list of `canonical_triples` to a JSON file (e.g., `canonical_triples.json`).
    *   Saves the schema information (including the `relation_definitions` map from Step 5.2 and the `canonical_map` from Step 5.6, potentially also the `canonical_entity_map` from Step 6.5) into a combined JSON file (e.g., `schema_info.json`).
    *   Optionally saves the initial text chunks generated in Phase 2 (e.g., `source_chunks.json`).
    *   These files are typically organized into a directory specific to the processed document/company (e.g., `companies/<company_name>/`).
*   **Why:** Persists the key data outputs for later use, debugging, or auditing. The canonical triples are the direct source for the graph. Schema info documents the standardization process. Source chunks can help trace facts back to the original text.
*   **Advantages:** Makes results reproducible and traceable. Allows the UI or other processes to load processed data without rerunning the entire pipeline (caching).
*   **Example:** Files like `companies/ExampleCorp/canonical_triples.json`, `companies/ExampleCorp/schema_info.json`, `companies/ExampleCorp/knowledge_graph.graphml`, etc., are created.

## Phase 9: RAG Querying 

This phase describes how the Natural Language Querying feature, implemented in the Streamlit UI (`app.py`) and RAG service (`rag_service/retriever.py`), works.

### Step 9.1: Load Graph Data (on demand)

*   **What:** Loads the previously processed knowledge graph and supporting data when a user selects a company/document in the UI.
*   **How:** When a user selects a company, the application loads:
    *   The `networkx` graph from the saved GraphML file (`nx.read_graphml(...)`).
    *   The canonical triples and schema info from the JSON files.
    *   Result is cached using `@st.cache_data` to improve performance on repeated queries.
*   **Why:** To make the relevant knowledge graph available for querying without reprocessing.
*   **Advantages:** Fast loading of previously processed results with built-in caching.
*   **Example:** User selects "AAPL". The app loads `companies/AAPL/knowledge_graph.graphml` into a `networkx` object `G`.

### Step 9.2: User Query Input (UI)

*   **What:** The user types a question in natural language into a text input field in the Streamlit UI.
*   **How:** Standard `streamlit.text_input` widget with a submit button.
*   **Why:** Provides the interface for the user to ask questions.
*   **Advantages:** User-friendly interaction method.
*   **Example:** User types: "Who acquired Beta Inc?"

### Step 9.3: Retrieve Relevant Context (Graph Retrieval)

*   **What:** Identifies nodes and edges in the knowledge graph that are relevant to the user's query.
*   **How:** This is a key RAG step implemented in `rag_service/retriever.py`:
    1.  **Entity Extraction & Linking:** Identify entity names from the user query (e.g., "Beta Inc") and find corresponding node(s) in the graph using both exact and fuzzy matching.
    2.  **Subgraph Extraction:** Find the neighbors of the identified entity node(s) within a configurable depth (typically 1-2 hops). This retrieves triples directly connected to the entity.
    3.  **Fallback Strategies:** When direct entity matches aren't found, the system uses embedding-based semantic search to find relevant nodes.
    4.  **Context Prioritization:** Triples are ranked by relevance to the query and a subset is selected to fit within context limits.
*   **Why:** To narrow down the vast information in the graph to only the parts most likely to contain the answer to the user's specific question. This provides focused context for the LLM.
*   **Advantages:** Prevents the LLM from being overwhelmed by the entire graph. Grounds the LLM's response in the actual data extracted from the document. Multiple fallback strategies ensure relevant information is found even with inexact queries.
*   **Example:** For "Who acquired Beta Inc?", the system:
    1. Identifies "Beta Inc" as an entity in the query
    2. Finds the node `"beta inc."` in the graph (using fuzzy matching if needed)
    3. Retrieves its incoming edges, finding `("acme corp", "beta inc.", label="acquire")`
    4. Formats this as context: "Acme Corp acquired Beta Inc."

### Step 9.4: Augment LLM Prompt

*   **What:** Constructs a new prompt for the LLM that includes both the original user query and the relevant context retrieved from the graph.
*   **How:** Creates a prompt template with specific instructions to use only the provided context: `"Based on the following context from the knowledge graph:\n{retrieved_context}\n\nAnswer the question: {user_query}\n\nIf the context doesn't contain information to answer the question, say so - do NOT use outside knowledge."`
*   **Why:** Provides the LLM with both the question and the specific information needed to answer it accurately, based *only* on the graph data, with explicit instructions to avoid hallucination.
*   **Advantages:** Guides the LLM to generate factual answers grounded in the extracted knowledge rather than generating information not present in the document.
*   **Example:** Prompt becomes: `"Based on the following context from the knowledge graph:\nAcme Corp acquired Beta Inc.\n\nAnswer the question: Who acquired Beta Inc?"`

### Step 9.5: Generate Response (Gemini LLM)

*   **What:** Sends the augmented prompt to the Gemini LLM to generate the final answer.
*   **How:** Calls the `Models.llm.generate_response` function with the augmented prompt, expecting a string response.
*   **Why:** Leverages Gemini's natural language generation capabilities to synthesize a coherent answer from the provided context.
*   **Advantages:** Produces human-readable answers instead of just raw data. Gemini's understanding of complex relationships helps interpret the graph data appropriately.
*   **Example:** The LLM receives the augmented prompt and generates the answer: `"Based on the provided information, Acme Corp acquired Beta Inc."`

### Step 9.6: Display Response and Context (UI)

*   **What:** Shows the LLM-generated answer to the user in the Streamlit UI, with an option to view the graph context used.
*   **How:** Uses `streamlit.markdown` to display the final answer, with an expandable section showing the triples used as context.
*   **Why:** Presents the result of the query to the user, with transparency about the information sources.
*   **Advantages:** Completes the interactive query loop while allowing users to verify the sources of information.
*   **Example:** The answer appears below the user's query, with an expandable "View Graph Context Used" section showing the triples that informed the response. 

## Phase 10: Triple Evaluation (Using LLM-as-Judge)

This optional phase uses an LLM to evaluate the quality and correctness of the extracted triples against their source text chunks.

### Step 10.1: Sampling Triples

*   **What:** Selects a subset of triples for evaluation.
*   **How:** Two main strategies are supported:
    1.  **Total Random Sampling (`--total` parameter):** Randomly selects a fixed number of triples (e.g., 50, 100) from the *entire* set of extracted triples across all chunks. The specific chunk ID for each selected triple is retained.
    2.  **Per-Chunk Sampling (`--samples` parameter):** Randomly selects a fixed number of triples (e.g., 5) from *each individual chunk* that contains triples. The total number of evaluated triples will vary depending on how many chunks produced triples.
*   **Why:** Evaluating all triples can be time-consuming and expensive due to LLM calls. Sampling provides a representative assessment of quality. Total random sampling gives a fixed sample size for overall quality, while per-chunk sampling helps understand quality variance between different parts of the document.
*   **Advantages:** Provides a manageable subset for evaluation. Allows flexible control over the evaluation scope and cost.
*   **Example:** Using `--total 100` selects 100 triples randomly from the entire `canonical_triples.json` file. Using `--samples 5` selects 5 triples from chunk 0, 5 from chunk 1 (if they exist), etc.

### Step 10.2: LLM-Based Evaluation for Each Sampled Triple

*   **What:** For each sampled triple, uses an LLM to judge its correctness based *only* on the text of its original source chunk.
*   **How:**
    1.  For a sampled triple, retrieve its `source_chunk` ID.
    2.  Load the corresponding text content from the `source_chunks.json` file.
    3.  Construct a detailed prompt for the LLM (e.g., Gemini):
        *   Provide the `source_chunk` text as context.
        *   Provide the `subject`, `predicate`, and `object` of the triple.
        *   Instruct the LLM to act as a validator.
        *   Ask the LLM to classify the triple as `CORRECT`, `PARTIAL`, or `INCORRECT` based *strictly* on the provided source text.
        *   Ask for a brief `reasoning` for the classification.
        *   Ask for a `confidence` score (0-1).
    4.  Use the `Models.llm.generate_response` function, specifying a Pydantic model (`TripleEvaluation`) to structure the LLM's output (classification, reasoning, confidence).
*   **Why:** Leverages the LLM's understanding to assess semantic correctness beyond simple string matching. Constraining the evaluation to the source chunk ensures the triple is judged against the context it was extracted from.
*   **Advantages:** Provides a nuanced, semantic evaluation of triple quality. Can identify issues like hallucination, misinterpretation, or lack of direct support in the text.
*   **Example:** Given a triple `(Apple, headquarters_location, Cupertino)` and its source chunk text containing "Apple Inc. is headquartered in Cupertino, California", the LLM should classify it as `CORRECT` with high confidence.

### Step 10.3: Aggregation and Reporting

*   **What:** Collects the LLM evaluation results for all sampled triples and generates summary reports.
*   **How:**
    1.  Stores the evaluation results (classification, reasoning, confidence) alongside each sampled triple.
    2.  Calculates aggregate statistics: overall rates of CORRECT, PARTIAL, INCORRECT classifications, and average confidence score.
    3.  Saves the detailed results (each triple + its evaluation) to a CSV file (e.g., `eval/results/COMPANY_evaluation_random100.csv`).
    4.  Saves the aggregate statistics to a separate summary CSV file (e.g., `eval/results/COMPANY_evaluation_random100_summary.csv`).
    5.  Optionally saves the full results structure to a JSON file.
*   **Why:** Provides both detailed, per-triple feedback and high-level summary metrics to understand the overall quality of the knowledge graph extraction process.
*   **Advantages:** Offers clear, quantifiable metrics for quality assessment. CSV format allows for easy analysis and comparison across different runs or documents.
*   **Example:** The summary CSV might show: `correct_rate: 0.85`, `partial_rate: 0.10`, `incorrect_rate: 0.05`, `avg_confidence: 0.92`.

## Phase 11: Performance Optimizations & Production Considerations

This phase outlines key performance optimizations and production-ready aspects implemented throughout the system.

### Step 11.1: Result Caching & Persistence

*   **What:** The system implements strategic caching and persistence of intermediate results throughout the pipeline.
*   **How:** 
    1.  **Disk Caching:** Key outputs like normalized triples, canonical triples, and schema info are saved to JSON files in company-specific directories.
    2.  **Pipeline Resumption:** The UI (`app.py`) checks for cached normalized triples before reprocessing a document, allowing it to skip the expensive extraction and normalization steps.
    3.  **In-Memory Caching:** The Streamlit app uses `@st.cache_data` decorator for expensive operations like loading graph data.
*   **Why:** PDF processing and LLM calls are resource-intensive and time-consuming. Caching allows efficient reuse of processed data and enables rapid iteration on later pipeline stages without redoing earlier steps.
*   **Advantages:** Dramatically improves performance for repeated operations. Enables graceful recovery from failures in later stages. Allows users to close and reopen the application without losing work.
*   **Example:** If a user processes an Apple 10-K and then closes the app, the normalized triples file `companies/AAPL/normalized_triples.json` remains. When they reopen the app, they can immediately apply schema canonicalization or visualization without reprocessing the PDF.

### Step 11.2: Parallel Processing

*   **What:** The system uses parallel processing at several compute-intensive stages.
*   **How:** 
    1.  **Chunk Processing (Phase 3):** `parallel_process_chunks` uses `concurrent.futures` to distribute LLM triple extraction across multiple threads/processes (`batch_size=10, max_workers=4`).
    2.  **Schema Definition (Phase 5):** `parallel_schema_processing` similarly parallelizes the LLM-based relation definition generation.
*   **Why:** Sequential LLM API calls would result in very slow performance, especially for large documents with many chunks or unique relations.
*   **Advantages:** Reduces overall processing time significantly by utilizing multiple CPU cores and handling multiple LLM requests concurrently. The batch size parameter allows control over memory consumption.
*   **Example:** If a document has 40 chunks, with 4 workers and a batch size of 10, the system processes 4 chunks simultaneously, with each worker handling one chunk at a time from its assigned batch of 10.

### Step 11.3: Visualization Performance Optimizations

*   **What:** The system implements special handling for visualizing large knowledge graphs effectively in the browser.
*   **How:** 
    1.  **Degree-Based Filtering:** The UI implements a slider to filter nodes by minimum degree, allowing users to focus on the most connected entities.
    2.  **Subgraph Generation:** Instead of visualizing the entire graph (which can be overwhelming), the app generates and displays a manageable subgraph.
    3.  **Physics Configuration:** The `pyvis` visualization uses carefully tuned physics parameters for better layout of large graphs.
*   **Why:** Raw knowledge graphs can become very large and visually overwhelming, especially for comprehensive documents like 10-Ks. Browser-based visualization tools have performance limitations.
*   **Advantages:** Provides a responsive, interactive visualization even for large graphs. Focuses attention on the most important/connected entities. Improves user experience.
*   **Example:** For an Apple 10-K with hundreds or thousands of extracted entities, setting a minimum degree of 3 might focus the visualization on only the ~50 most connected entities (e.g., "Apple", "iPhone", "Tim Cook", "China", "Services"), making the visualization both more meaningful and performant.

### Step 11.4: Error Handling & Robustness

*   **What:** The system implements comprehensive error handling throughout the pipeline.
*   **How:** 
    1.  **Retry Logic:** LLM API calls include automatic retries with exponential backoff.
    2.  **Error Recording:** Failed chunks and relation definitions are systematically tracked rather than causing pipeline failure.
    3.  **Default Values:** Key functions include fallback return values when errors occur.
    4.  **UI Status Messages:** The Streamlit UI provides clear error messages and warnings.
*   **Why:** External API calls, especially to AI services, can be unpredictable. Documents can contain unexpected content. A production system needs to handle these gracefully.
*   **Advantages:** Improves overall pipeline reliability. Provides transparency into partial failures. Allows successful parts of the process to continue even when some elements fail.
*   **Example:** If the LLM extraction on chunk #23 fails due to an API timeout, the system will retry up to 3 times with increasing delays. If still unsuccessful, it records this as a failure but continues processing other chunks. 

## Appendix: Concrete Example - Processing Apple's 10-K

This appendix walks through the entire process as applied to a real Apple Inc. 10-K filing, illustrating how each phase works in practice.

### Document Overview

A 10-K filing for Apple Inc. (AAPL) is a comprehensive annual report containing:
- Business description (products, services, markets)
- Risk factors
- Management's discussion and analysis (MD&A)
- Financial statements and supplementary data
- Information about executives and governance
- Exhibits and subsidiary information

### Phase 1: Extraction Example

1. **PDF Loading:** The system loads `AAPL_10K.pdf` using `pdfplumber`.
2. **Table Extraction:** Financial tables are detected, such as:
   ```
   --- TABLE 24-1 ---
   (In millions)     | 2023    | 2022    | 2021
   Net sales         | $394,328| $394,328| $365,817
   Operating income  | $114,300| $119,437| $108,949
   --- END TABLE 24-1 ---
   ```
3. **Full Text:** The system extracts text like:
   ```
   Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services.
   
   The Company's products include iPhone, Mac, iPad, AirPods, Apple TV, Apple Watch, Beats products, HomePod, iPod touch and accessories.
   ```

### Phase 2: Chunking Example

1. **Text Splitting:** The document might be split into ~100 chunks (depending on size).
2. **Table Handling:** Large financial tables are processed separately, with custom chunk sizes.
3. **Example Chunk:** A chunk might contain text like:
   ```
   The Company's fiscal year is the 52-or 53-week period that ends on the last Saturday of September. The Company's fiscal years 2023, 2022 and 2021 spanned 52 weeks each. Unless otherwise stated, references to particular years, quarters, months and periods refer to the Company's fiscal years ended in September and the associated quarters, months and periods of those fiscal years.
   
   Net sales decreased during 2023 compared to 2022 due primarily to lower net sales of iPhone, Mac and iPad, partially offset by higher net sales of Services. The weakness in foreign currencies against the U.S. dollar negatively impacted net sales during 2023 compared to 2022.
   ```

### Phase 3: Triple Extraction Example

1. **LLM Extraction (Gemini):** For the chunk above, the LLM might extract triples like:
   ```
   {"subject": "Apple Inc", "predicate": "has fiscal year ending", "object": "last Saturday of September"}
   {"subject": "Apple's fiscal years 2023, 2022, and 2021", "predicate": "spanned", "object": "52 weeks each"}
   {"subject": "Net sales", "predicate": "decreased during", "object": "2023 compared to 2022"}
   {"subject": "Lower net sales of iPhone, Mac and iPad", "predicate": "caused", "object": "decreased net sales in 2023"}
   {"subject": "Higher net sales of Services", "predicate": "partially offset", "object": "decreased net sales in 2023"}
   {"subject": "Weakness in foreign currencies against the U.S. dollar", "predicate": "negatively impacted", "object": "net sales during 2023"}
   ```

2. **Error Handling:** If a complex financial table causes an LLM timeout, the system retries and if still unsuccessful, adds it to the `failures` list.

### Phase 4: Normalization Example

1. **Lemmatization:** The triples are normalized:
   ```
   {"subject": "apple inc", "predicate": "have fiscal year end", "object": "last saturday of september"}
   {"subject": "apple fiscal year 2023 2022 and 2021", "predicate": "span", "object": "52 week each"}
   {"subject": "net sale", "predicate": "decrease during", "object": "2023 compare to 2022"}
   ```

2. **Deduplication:** If similar triples are extracted from different parts of the document (e.g., from the business overview and from MD&A), duplicates are removed.

### Phase 5: Schema Canonicalization Example

1. **Relation Definitions (Gemini):** The LLM generates definitions like:
   ```
   "have fiscal year end": "Specifies the date on which a company's fiscal year concludes."
   "manufacture": "To produce or create goods through industrial processes."
   "design": "To conceive and plan the form and structure of a product."
   "sell": "To offer goods or services in exchange for money."
   "decrease during": "To become smaller in size, amount, or degree over a specified period."
   ```

2. **Similarity Calculation (BGE):** The system might determine that "sell", "offer", and "market" have similar definition embeddings.

3. **Canonicalization:** These are grouped, with "sell" chosen as the canonical relation:
   ```
   "sell" -> "sell"
   "offer" -> "sell" 
   "market" -> "sell"
   ```

### Phase 6: Entity Canonicalization Example

1. **Entity Filtering:** Entities are filtered to exclude those with digits:
   - Included: "apple inc", "iphone", "tim cook", "united states"
   - Excluded: "2023", "52 weeks", "$394,328", "10%"

2. **Similarity Calculation (BGE):** The system finds that:
   - "apple inc" and "apple" are highly similar
   - "iphone" and "iphones" are highly similar
   - "united states", "u.s.", and "usa" are highly similar

3. **Canonicalization Result:**
   ```
   "apple" -> "apple inc"
   "iphones" -> "iphone"
   "u.s." -> "united states" 
   "usa" -> "united states"
   ```

### Phase 7-8: Graph Construction & Visualization Example

1. **Graph Building:** The system creates a NetworkX graph with ~1000 nodes (entities) and ~3000 edges (relationships).
   - Major entities like "apple inc", "iphone", "services", "tim cook" have many connections
   - Financial figures, subsidiaries, and product details form distinct clusters

2. **Visualization:** The resulting HTML visualization shows:
   - "apple inc" as a central hub node
   - Product clusters (iPhone, Mac, iPad, Services)
   - Geographic clusters (United States, China, Europe)
   - Executive/governance clusters
   - Financial performance relationships

### Phase 9: RAG Query Example

1. **User Query:** "What caused Apple's net sales to decrease in 2023?"

2. **Entity Detection:** The system identifies "Apple" and "2023" as entities in the query.

3. **Context Retrieval:** The system finds triples mentioning Apple/net sales and 2023:
   ```
   (net sale, decrease during, 2023 compare to 2022)
   (lower net sale of iphone mac and ipad, cause, decrease net sale in 2023)
   (higher net sale of service, partially offset, decrease net sale in 2023)
   (weakness in foreign currency against the u.s. dollar, negatively impact, net sale during 2023)
   ```

4. **LLM Response (Gemini):** "According to the information, Apple's net sales decreased in 2023 compared to 2022 primarily due to lower sales of iPhone, Mac, and iPad. This decrease was partially offset by higher net sales of Services. Additionally, the weakness of foreign currencies against the U.S. dollar negatively impacted net sales during 2023."

### Phase 10: Triple Evaluation Example

1. **Sampling:** Using `--total 100`, 100 random triples are selected from the ~3000 canonical triples for AAPL.
2. **LLM Evaluation:** The triple `(weakness in foreign currency against the u.s. dollar, negatively impact, net sale during 2023)` is evaluated against its source chunk text.
3. **Result:** The LLM classifies it as `CORRECT`, with reasoning like "The text explicitly states that currency weakness negatively impacted net sales during 2023." and a confidence of 0.95.
4. **Reporting:** A file `eval/results/AAPL_evaluation_random100.csv` is generated, and the summary might show `correct_rate: 0.90`, `partial_rate: 0.08`, `incorrect_rate: 0.02`.

### Phase 11: Performance Optimizations Example

1. **Caching:** After initial processing, `companies/AAPL/normalized_triples.json` contains ~3000 extracted and normalized triples, allowing the system to skip extraction on subsequent runs.

2. **Parallel Processing:** With `max_workers=4`, the system processes 4 document chunks simultaneously, extracting triples from different sections of the 10-K concurrently.

3. **Visualization Performance:** For the visualization, setting minimum degree=3 might reduce the displayed graph from 1000+ nodes to a more manageable ~80 key entities, focusing on the most significant relationships. 