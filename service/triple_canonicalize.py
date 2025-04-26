from Models.llm import generate_response
from Models.embed import get_embeddings_batch, compute_similarity
from typing import List, Tuple, Dict
import math
import concurrent.futures
from pydantic import BaseModel, Field
import re 

# Helper function for normalization
def normalize_entity_string(text: str) -> str:
    if not isinstance(text, str):
        return text 
    text = text.lower()
    text = text.strip()
    text = text.replace(",", "") 
    return text

class RelationDefinitionPair(BaseModel):
    relation: str
    definition: str

class RelationDefinitionsListModel(BaseModel):
    definitions: List[RelationDefinitionPair]

class RelationDefinitionsBatch(BaseModel):
    definitions: Dict[str, str]

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
            prompt = f"""Given the relation '{relation}' extracted from a company's financial documents,
            provide a clear, concise one-sentence definition of what this relation means in the context
            of business and financial reporting. Focus on precision and clarity."""
            
            response = generate_response(prompt, str)
            
            relation_definitions[relation] = response.strip()
            #print(f"✓ Defined: {relation}")
            
        except Exception as e:
            print(f"✗ Failed to define '{relation}': {str(e)}")
            relation_definitions[relation] = f"Undefined relation: {relation}"
    
    print(f"Schema definition complete. Defined {len(relation_definitions)} relations.")
    return relation_definitions



def canonicalize_schema(triples: List[dict], relation_definitions: Dict[str, str], 
                        similarity_threshold: float = 0.90) -> Tuple[List[dict], Dict[str, str]]:
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
        
        print("Generating embeddings...")
        embeddings = get_embeddings_batch(definitions, "")
        
        canonical_map = {}
        processed = set()
        
        all_scores = []
        
        print("\nFinding similar relations...")
        for i, relation in enumerate(relation_keys):
            if relation in processed:
                continue
                
            similar_relations = []
            
            # Find similar relations
            for j, other_relation in enumerate(relation_keys):
                if i != j and other_relation not in processed:
                    similarity = compute_similarity(embeddings[i], embeddings[j])
                    
                    all_scores.append({
                        'relation1': relation,
                        'relation2': other_relation,
                        'similarity': similarity
                    })
                    
                    if similarity > similarity_threshold:
                        similar_relations.append((other_relation, similarity))
            
            if similar_relations:
                similar_relations.sort(key=lambda x: x[1], reverse=True)
                
                canonical_relation = relation
                processed.add(canonical_relation)
                canonical_map[canonical_relation] = canonical_relation
                
                # Map similar relations to the canonical form
                for similar_rel, score in similar_relations:
                    canonical_map[similar_rel] = canonical_relation
                    processed.add(similar_rel)
                    print(f"Mapped '{similar_rel}' → '{canonical_relation}' (similarity: {score:.2f})")
            else:
                canonical_map[relation] = relation
                processed.add(relation)
        
        print("\nUpdating triples with canonical relations...")
        canonical_triples = []
        for triple in triples:
            canonical_triple = triple.copy()
            canonical_triple['predicate'] = canonical_map[triple['predicate']]
            canonical_triples.append(canonical_triple)
        
        original_relations = len(set(t['predicate'] for t in triples))
        canonical_relations = len(set(t['predicate'] for t in canonical_triples))
        
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
    


def parallel_schema_processing(normalized_triples: List[dict], batch_size: int = 200, max_workers: int = 4) -> Tuple[Dict[str, str], List[str]]:
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
    #print(f"DEBUG: Using batch_size = {batch_size}, max_workers = {max_workers}")
    
    def process_relation_batch(relations_batch: List[str]) -> Tuple[Dict[str, str], List[str]]:
        batch_definitions = {}
        batch_failures = []
        
        if not relations_batch:
            return {}, []

        
        for relation in relations_batch:
            try:
                prompt = f"""Given the relation '{relation}' extracted from a company's financial documents,
                provide a clear, concise one-sentence definition of what this relation means in the context
                of business and financial reporting. Focus on precision and clarity."""
                
                response = generate_response(prompt, str).strip()
                
                if response:
                     batch_definitions[relation] = response
                     #print(f"✓ Defined: {relation}")
                else:
                    print(f"✗ Empty response received for relation '{relation}'. Marking as failed.")
                    batch_failures.append(relation)
                    batch_definitions[relation] = f"Definition not generated for: {relation}" # Assign default on empty response

            except Exception as e:
                print(f"✗ Failed to define '{relation}': {str(e)}")
                batch_failures.append(relation)
                batch_definitions[relation] = f"Definition not generated for: {relation}" # Ensure it's added even on failure
        
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

def canonicalize_entities(triples: List[dict], similarity_threshold: float = 0.90) -> Tuple[List[dict], Dict[str, str]]:
    """
    Canonicalize entities in knowledge graph triples by finding similar entity names using embeddings.
    
    Args:
        triples: List of knowledge graph triples
        similarity_threshold: Threshold for similarity (default: 0.85)
        
    Returns:
        Tuple of (canonicalized triples, mapping of original to canonical entities)
    """
    print(f"\nPhase 4: Canonicalizing Entity Names (Similarity Threshold: {similarity_threshold})...")
    
    original_unique_entities_set = set()
    try:
        for triple in triples:
            if isinstance(triple['subject'], str):
                 original_unique_entities_set.add(triple['subject'])
            if isinstance(triple['object'], str):
                 original_unique_entities_set.add(triple['object'])
        
        all_unique_entities = list(original_unique_entities_set)
        print(f"Found {len(all_unique_entities)} total unique string entities")

        # --- Filtering Step --- 
        # Check if ANY digit exists in the normalized string
        
        entities_to_canonicalize = []
        skipped_entities = []
        entities_not_processed = [] 

        print("Filtering entities: Skipping any containing digits...")
        for entity in all_unique_entities:
            if isinstance(entity, str):
                normalized_entity = normalize_entity_string(entity)
                if re.search(r'\d', normalized_entity):
                    skipped_entities.append(entity) 
                else:
                    entities_to_canonicalize.append(entity) 
            else:
                 skipped_entities.append(entity)
                 entities_not_processed.append(entity)
        
        if entities_not_processed:
             print(f"- Skipped {len(entities_not_processed)} non-string entities.")

        print(f"- Identified {len(entities_to_canonicalize)} string entities for semantic canonicalization (no digits detected).")
        print(f"- Skipped {len(skipped_entities) - len(entities_not_processed)} entities containing digits.")
        # --- End Filtering Step --- 

        if not entities_to_canonicalize:
             print("No entities identified for semantic canonicalization. Skipping similarity checks.")
             canonical_map = {entity: entity for entity in all_unique_entities}
        else:
            # Generate embeddings ONLY for entities_to_canonicalize
            print("Generating embeddings for potential named entities...")
            embeddings = get_embeddings_batch(entities_to_canonicalize, "")
            
            canonical_map = {entity: entity for entity in skipped_entities}
            processed = set(skipped_entities) 
            
            all_scores = []
            
            print("\nFinding similar named entities...")
            for i, entity in enumerate(entities_to_canonicalize):
                if entity in processed:
                    continue
                    
                similar_entities = []
                
                # Find similar entities within the candidates
                for j, other_entity in enumerate(entities_to_canonicalize):
                    # Skip self-comparison and already processed entities
                    if i != j and other_entity not in processed:
                        similarity = compute_similarity(embeddings[i], embeddings[j])
                        
                        # Track scores for analysis
                        all_scores.append({
                            'entity1': entity,
                            'entity2': other_entity,
                            'similarity': similarity
                        })
                        
                        if similarity > similarity_threshold:
                            similar_entities.append((other_entity, similarity))
                
                if similar_entities:
                    similar_entities.sort(key=lambda x: x[1], reverse=True)
                    
                    canonical_entity = entity
                    processed.add(canonical_entity)
                    canonical_map[canonical_entity] = canonical_entity
                    
                    for similar_entity, score in similar_entities:
                        if similar_entity not in canonical_map: 
                            canonical_map[similar_entity] = canonical_entity
                            processed.add(similar_entity)
                            print(f"Mapped entity '{similar_entity}' → '{canonical_entity}' (similarity: {score:.2f})")
                else:
                    if entity not in canonical_map:
                        canonical_map[entity] = entity
                        processed.add(entity)
            
            for entity in entities_to_canonicalize:
                if entity not in canonical_map:
                    canonical_map[entity] = entity

        print("\nUpdating triples with canonical entities...")
        canonical_triples = []
        for entity in all_unique_entities:
             if entity not in canonical_map:
                  canonical_map[entity] = entity 

        for triple in triples:
            canonical_triple = triple.copy()
            subj = triple['subject']
            obj = triple['object']
            canonical_triple['subject'] = canonical_map.get(subj, subj) 
            canonical_triple['object'] = canonical_map.get(obj, obj)  
            canonical_triples.append(canonical_triple)
        
        original_entities_count = len(all_unique_entities)
        canonical_entities_count = len(set(canonical_map.values()))
        entities_merged = original_entities_count - canonical_entities_count
        
        print(f"\nEntity canonicalization complete:")
        print(f"- Original unique entities: {original_entities_count}")
        print(f"- Canonical unique entities: {canonical_entities_count}")
        print(f"- Entities merged (semantic): {entities_merged}")
        
        return canonical_triples, canonical_map
        
    except Exception as e:
        print(f"Error during entity canonicalization: {str(e)}")
        return triples, {entity: entity for entity in original_unique_entities_set}