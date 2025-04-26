import networkx as nx
import os
from typing import List, Dict, Tuple, Optional
from utils.prompts import entity_prompt, qa_prompt
from Models.llm import generate_response
from rapidfuzz import process, fuzz

def link_entities_to_graph(question: str, G: nx.Graph, llm_based_extraction: bool = True, fuzzy_threshold: int = 85) -> List[str]:
    """
    Extracts entities from a question using an LLM and finds corresponding nodes in the graph.
    Uses rapidfuzz for fuzzy matching as a fallback or primary method if LLM extraction is disabled.
    
    Args:
        question: The user's natural language question.
        G: The loaded knowledge graph.
        llm_based_extraction: Whether to use LLM for entity extraction.
        fuzzy_threshold: Minimum similarity score (0-100) for a fuzzy match.
        
    Returns:
        A list of unique node names found in the graph that correspond to entities in the question.
    """
    print(f"Linking entities for question: '{question}'")
    potential_entities = []
    
    if llm_based_extraction:
        try:
            prompt = entity_prompt.format(question=question)
            
            entity_string = generate_response(prompt, str)
            
            potential_entities = [entity.strip().lower() for entity in entity_string.split(',') if entity.strip()]
            print(f"LLM extracted potential entities: {potential_entities}")
        except Exception as e:
            print(f"LLM entity extraction failed: {str(e)}. Falling back to basic extraction.")
            llm_based_extraction = False 
            
    if not llm_based_extraction or not potential_entities:
        
        potential_entities = question.lower().split()
        print(f"Using basic tokenization for potential entities: {potential_entities}")

    found_nodes = set()
    graph_nodes = [str(node) for node in G.nodes()]
    all_nodes_lower_map = {node.lower(): node for node in graph_nodes}
    lowercase_node_list = list(all_nodes_lower_map.keys())

    if not lowercase_node_list: 
        print("Graph has no nodes to match against.")
        return []

    for entity in potential_entities:
        if not entity: continue 

        if entity in all_nodes_lower_map:
            found_nodes.add(all_nodes_lower_map[entity])
            print(f"Exact matched '{entity}' to node '{all_nodes_lower_map[entity]}'")
            continue 

        match = process.extractOne(entity, lowercase_node_list, scorer=fuzz.WRatio, score_cutoff=fuzzy_threshold)
        
        if match:
            matched_node_lower = match[0]
            matched_node_original = all_nodes_lower_map[matched_node_lower]
            if matched_node_original not in found_nodes:
                 found_nodes.add(matched_node_original)
                 print(f"Fuzzy matched '{entity}' to node '{matched_node_original}' with score {match[1]:.2f}")
        else:
           for node_lower, node_original in all_nodes_lower_map.items():
               if entity in node_lower and node_original not in found_nodes:
                   found_nodes.add(node_original)
                   print(f"Substring matched '{entity}' to node '{node_original}'")
            
    print(f"Found corresponding graph nodes: {list(found_nodes)}")
    return list(found_nodes)


def retrieve_graph_context(start_nodes: List[str], G: nx.Graph, depth: int = 2, max_triples: int = 550) -> str:
    """
    Retrieves context from the graph around start nodes by exploring neighbors up to a specified depth.
    Limits the total number of triples returned.
    
    Args:
        start_nodes: List of nodes in the graph to start traversal from.
        G: The loaded knowledge graph.
        depth: How many hops to explore (default: 2).
        max_triples: Maximum number of triples to include in the context (default: 100).
        
    Returns:
        A formatted string representing the retrieved graph context, or empty string if none.
    """
    if not start_nodes:
        return ""
        
    print(f"Retrieving context for nodes: {start_nodes} with depth {depth}, max triples {max_triples}")
    context_triples = set()
    nodes_to_explore = set(start_nodes)
    visited_nodes = set() 
    all_nodes_in_context = set(start_nodes) 

    for current_depth in range(depth):
        next_nodes_to_explore = set()
        if not nodes_to_explore:
            break
            
        current_batch = list(nodes_to_explore) 
        visited_nodes.update(current_batch)
        
        print(f"Depth {current_depth + 1}: Exploring {len(current_batch)} nodes...")

        for node in current_batch:
            if node not in G:
                continue
                
            try:
                neighbors = list(G.neighbors(node))
            except Exception as e:
                 print(f"Warning: Could not get neighbors for node {node}: {e}")
                 continue
                 
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data is None: continue # Skip if edge data is missing
                
                relation = edge_data.get('edge', edge_data.get('relation', 'related_to')) 
                context_triples.add(f"({node}) -> [{relation}] -> ({neighbor})")
                all_nodes_in_context.add(node)
                all_nodes_in_context.add(neighbor)
                if neighbor not in visited_nodes and neighbor not in nodes_to_explore:
                    next_nodes_to_explore.add(neighbor)
            
            if G.is_directed():
                try:
                    predecessors = list(G.predecessors(node))
                except Exception as e:
                    print(f"Warning: Could not get predecessors for node {node}: {e}")
                    continue
                    
                for predecessor in predecessors:
                    if predecessor == node: continue 

                    edge_data = G.get_edge_data(predecessor, node)
                    if edge_data is None: continue
                    
                    relation = edge_data.get('edge', edge_data.get('relation', 'related_to'))
                    context_triples.add(f"({predecessor}) -> [{relation}] -> ({node})")
                    all_nodes_in_context.add(predecessor)
                    all_nodes_in_context.add(node)
                    if predecessor not in visited_nodes and predecessor not in nodes_to_explore:
                         next_nodes_to_explore.add(predecessor)
                         
        nodes_to_explore = next_nodes_to_explore

    if not context_triples:
        print("No context triples found within the specified depth.")
        return f"Entities found in graph: {', '.join(start_nodes)}" 
        
    # Apply limit
    final_triples = sorted(list(context_triples))
    if len(final_triples) > max_triples:
        print(f"Context exceeded limit ({max_triples}). Truncating {len(final_triples)} triples...")
        final_triples = final_triples[:max_triples]
        
    formatted_context = "\n".join(final_triples)
    print(f"Retrieved context ({len(final_triples)} triples):")
    return formatted_context

def query_graph_rag(question: str, company_name: str) -> Tuple[str, str]:
    """
    Main function to perform Graph RAG.
    Loads graph, links entities, retrieves context (default 2-hop), and calls LLM for answer.
    
    Args:
        question: The user's natural language question.
        company_name: The name of the company whose graph should be queried.
        
    Returns:
        A tuple containing:
            - The answer generated by the LLM or an error/info message.
            - The graph context string used for generation (or an empty string if none/error).
    """
    graph_context = "" 
    graph_path = os.path.join("companies", company_name, "graph_output", "knowledge_graph.graphml")
    if not os.path.exists(graph_path):
        return f"Error: Knowledge graph for company '{company_name}' not found at {graph_path}", graph_context
    
    try:
        G = nx.read_graphml(graph_path)
        print(f"Successfully loaded graph for {company_name} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except Exception as e:
        return f"Error loading graph for company '{company_name}': {str(e)}", graph_context

    # 2. Link Entities
    linked_nodes = link_entities_to_graph(question, G)
    if not linked_nodes:
        return "Could not link any entities from the question to the knowledge graph. Unable to retrieve graph context.", graph_context

    # 3. Retrieve Context
    graph_context = retrieve_graph_context(linked_nodes, G) 
    if not graph_context or graph_context.startswith("Entities found in graph:"):
         print(f"Found relevant entities ({linked_nodes}), but no actual context triples were retrieved within 2 hops.")
         if not graph_context: 
              graph_context = f"Entities found in graph: {', '.join(linked_nodes)}"

    # 4. Augmented Generation (LLM Call)
    print("Sending context and question to LLM for final answer generation...")
    try:
        formatted_prompt = qa_prompt.format(context=graph_context, question=question)
        
        answer = generate_response(formatted_prompt, str)
        
        print(f"LLM generated answer: {answer}")
        return answer, graph_context 
        
    except Exception as e:
        error_msg = f"Error generating final answer using LLM: {str(e)}"
        print(error_msg)
        return error_msg, graph_context 