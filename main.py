from utils.pdf_splitter import extract_text_from_pdf, chunk_text_with_tables
from service.triples_generation import parallel_process_chunks, normalize_and_deduplicate_triples
from service.triple_canonicalize import canonicalize_schema, parallel_schema_processing, canonicalize_entities

from service.visulization import visualize_chunks, build_and_visualize_knowledge_graph
from utils.prompts import extraction_system_prompt
from utils.graph_builder import create_graph, query_sub_graph
from Models.embed import get_embeddings_batch
from Models.llm import generate_response
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
import networkx as nx
from pathlib import Path


def main(args):
    company_dir = os.path.join("companies", args.company_name)
    os.makedirs(company_dir, exist_ok=True)
    
    output_dir = os.path.join(company_dir, "graph_output")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.pdf_path:
        print(f"Extracting text from: {args.pdf_path}")
        text = extract_text_from_pdf(args.pdf_path)
        print("Chunking text...")
        chunks = chunk_text_with_tables(text)
        print(f"Total chunks: {len(chunks)}")
        
        print("Extracting triples from chunks...")
        all_triples, failures = parallel_process_chunks(
            chunks, 
            extraction_system_prompt,
            args.company_name,
            batch_size=args.batch_size,
            max_workers=args.workers
        )
        print(f"Extracted {len(all_triples)} raw triples with {len(failures)} failures")
        
        print("Normalizing and deduplicating triples...")
        normalized_triples = normalize_and_deduplicate_triples(all_triples)
        print(f"Normalized to {len(normalized_triples)} unique triples")
        
        with open(os.path.join(company_dir, 'normalized_triples.json'), 'w') as f:
            json.dump(normalized_triples, f, indent=2)
        
    elif os.path.exists(os.path.join(company_dir, 'normalized_triples.json')):
        print(f"Loading normalized triples from: {os.path.join(company_dir, 'normalized_triples.json')}")
        with open(os.path.join(company_dir, 'normalized_triples.json'), 'r') as f:
            normalized_triples = json.load(f)
        print(f"Loaded {len(normalized_triples)} normalized triples")
    else:
        print("Error: Please provide a PDF file or ensure normalized triples exist.")
        return
    
    # Define schema for relations
    print("Defining schema...")
    relation_definitions, failed_relations = parallel_schema_processing(
        normalized_triples, 
        max_workers=args.workers
    )
    print(f"Defined {len(relation_definitions)} relations with {len(failed_relations)} failures")
    
    # Canonicalize schema
    print("Canonicalizing schema...")
    canonical_triples, canonical_relation_map = canonicalize_schema(
        normalized_triples,
        relation_definitions
    )
    print(f"Canonicalized to {len(canonical_triples)} triples")
    
    # Canonicalize entities
    print("Canonicalizing entities...")
    canonical_triples, canonical_entity_map = canonicalize_entities(
        canonical_triples,
        similarity_threshold=args.similarity_threshold
    )
    print(f"Entity canonicalization complete")
    
    canonical_triples_path = os.path.join(company_dir, 'canonical_triples.json')
    with open(canonical_triples_path, 'w') as f:
        json.dump(canonical_triples, f, indent=2)
    print(f"Saved canonical triples to {canonical_triples_path}")
    
    schema_info = {
        'relation_definitions': relation_definitions,
        'relation_mapping': canonical_relation_map,
        'entity_mapping': canonical_entity_map,
        'failed_relations': failed_relations
    }
    schema_info_path = os.path.join(company_dir, 'schema_info.json')
    with open(schema_info_path, 'w') as f:
        json.dump(schema_info, f, indent=2)
    print(f"Saved schema information to {schema_info_path}")
    
    triples_df = pd.DataFrame(canonical_triples)
    if 'subject' in triples_df.columns and 'object' in triples_df.columns:
        triples_df = triples_df.rename(columns={'subject': 'node_1', 'object': 'node_2', 'predicate': 'edge'})
    elif 'head' in triples_df.columns and 'tail' in triples_df.columns:
        triples_df = triples_df.rename(columns={'head': 'node_1', 'tail': 'node_2', 'relation': 'edge'})
    
    # Build and visualize knowledge graph
    print("Building and visualizing knowledge graph...")
    G, graphml_path = build_and_visualize_knowledge_graph(triples_df, output_dir)
    print(f"Knowledge graph built and visualization saved")
    print(f"GraphML file saved to: {graphml_path}")
    print(f"HTML visualization saved to: {os.path.join(output_dir, 'knowledge_graph_interactive.html')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF document to create a knowledge graph.")
    parser.add_argument("--company_name", required=True, help="Name of the company to process")
    parser.add_argument("--pdf_path", help="Path to the PDF document to process")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for chunked processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--similarity_threshold", type=float, default=0.85, help="Similarity threshold for entity canonicalization")
    args = parser.parse_args()
    
    main(args)

    