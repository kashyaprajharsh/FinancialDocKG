# Automated Knowledge Graph Builder from PDFs

This project extracts factual knowledge triples from PDF documents, particularly financial filings like 10-Ks, builds a knowledge graph, and provides an interactive interface for visualization and querying.

It leverages Large Language Models (LLMs) and Natural Language Processing (NLP) techniques to automatically structure information found in text and tables, handling challenges like semantic ambiguity and data redundancy.

## Key Features

*   **PDF Text/Table Extraction:** Uses `pdfplumber` to accurately extract both running text and structured tables from complex PDF layouts.
*   **Table-Aware Chunking:** Intelligently splits documents into manageable chunks for LLM processing, preserving table integrity.
*   **LLM-Powered Triple Extraction:** Uses Google's Gemini API to identify and extract Subject-Predicate-Object knowledge triples from text chunks.
*   **Semantic Canonicalization:** Employs BGE embedding models and cosine similarity to identify and merge synonymous relations (predicates) and entities (subjects/objects) based on semantic meaning, creating a clean, consistent graph schema.
*   **Entity Filtering:** Excludes numerical entities (dates, figures) from semantic canonicalization to improve accuracy.
*   **Robust Processing:** Includes parallel processing for performance, automatic retries for API calls, and systematic error tracking.
*   **Interactive Visualization:** Generates an interactive graph visualization using `pyvis` and `NetworkX`, allowing exploration of the extracted knowledge.
*   **RAG Querying:** Implements a Retrieval-Augmented Generation (RAG) system allowing users to ask natural language questions against the generated knowledge graph, grounding answers in document facts.
*   **Caching:** Persists intermediate results (normalized triples, schema info, final graphs) to disk, enabling faster reprocessing and analysis.
*   **Graph Building & Visualization**
*   **(Note: Intermediate results are cached in the `companies/<company_name>/` directory, speeding up subsequent runs.)*

## Technology Stack

*   **Python 3.11**
*   **LLM:** Google Gemini API
*   **Embeddings:** BGE (BAAI General Embedding) Models
*   **PDF Parsing:** `pdfplumber`
*   **NLP:** `nltk` (for lemmatization)
*   **Graph:** `networkx`
*   **Visualization:** `pyvis`, `streamlit-agraph` 
*   **Web Framework:** `streamlit`
*   **Core Libraries:** `pandas`, `numpy`, `concurrent.futures`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key Configuration:**
    *   This project requires access to the Google Gemini API.
    *   Create a file named `.env` in the root directory of the project.
    *   Add your API key to the `.env` file like this:
      ```
      GOOGLE_API_KEY=your_actual_google_api_key
      ```
    *   You can optionally specify the Gemini model to use (defaults to `gemini-1.5-flash`):
      ```
      GEMINI_MODEL_NAME=gemini-1.5-pro
      ```
    *   Replace `your_actual_google_api_key` with your real key.
    *   The `.env` file is included in `.gitignore` and should **never** be committed to version control.
    *   Refer to Google's documentation for obtaining API keys if you don't have one.

5.  **NLTK Data:**
    *   The first time you run the application, it should automatically download necessary `nltk` data (`punkt`, `wordnet`, `averaged_perceptron_tagger`). If you encounter issues, you can manually download them by running Python and entering:
      ```python
      import nltk
      nltk.download('punkt')
      nltk.download('wordnet')
      nltk.download('averaged_perceptron_tagger')
      ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Upload PDF:** Use the sidebar to upload a PDF document (e.g., a 10-K filing). Ensure the filename format allows for company name extraction (e.g., `AAPL_10K_2023.pdf`).

3.  **Processing:** The application will show progress through the different phases:
    *   Text Extraction & Chunking
    *   Triple Extraction (using LLM)
    *   Normalization & Deduplication
    *   Schema & Entity Canonicalization
    *   Graph Building & Visualization
    *   *(Note: Intermediate results are cached in the `companies/<company_name>/` directory, speeding up subsequent runs.)*

4.  **Explore:** Once processing is complete, use the tabs to:
    *   **Visualize:** View and interact with the knowledge graph (filter by node degree).
    *   **Explore Triples:** Browse, search, and export the extracted canonical triples.
    *   **Schema Info:** Examine the generated relation definitions and canonical mappings.
    *   **Query (RAG):** Ask natural language questions about the document content.

5.  **Evaluate Extracted Triples (Optional):**
    *   Use the LLM-based evaluation module to assess the quality of extracted triples.
    *   **Evaluate a specific company (sampling 50 random triples):**
        ```bash
        python -m eval.triple_evaluator --company <COMPANY_NAME> --total 50
        ```
    *   **Evaluate all companies:**
        ```bash
        python -m eval.evaluate_all_companies --total 50
        ```
    *   Replace `<COMPANY_NAME>` with the actual company directory name (e.g., `APPLE`).
    *   Adjust `--total` to change the number of randomly sampled triples.
    *   Results are saved as CSV and JSON files in the `eval/results/` directory.

## Project Structure

```
.
├── .gitignore         # Files ignored by Git
├── app.py             # Main Streamlit application
├── DETAILED_PROCESS_DOC.md # In-depth documentation of the pipeline
├── main.py            # Potential CLI entry point (if developed)
├── requirements.txt   # Project dependencies
├── README.md          # This file
├── companies/         # Output directory for processed data (ignored by git)
├── eval/              # Triple evaluation scripts and results
├── Models/            # LLM and Embedding model interaction logic
├── rag_service/       # RAG retrieval and querying logic
├── schemas/           # Pydantic models for data structures
├── service/           # Core processing logic for pipeline phases
└── utils/             # Utility functions (PDF splitting, prompts, graph building)
```

## Detailed Documentation

For a step-by-step explanation of each phase, the algorithms used, and design decisions, please refer to the [DETAILED_PROCESS_DOC.md](./DETAILED_PROCESS_DOC.md) file. 