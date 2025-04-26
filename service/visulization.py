import pandas as pd
from IPython.display import display
import os
import json
import networkx as nx
from pyvis.network import Network
from utils.graph_builder import create_graph

def visualize_chunks(chunks):
    print("--- Chunk Details ---")
    if chunks:
        chunks_data = [{"chunk_number": i+1, "text": chunk} for i, chunk in enumerate(chunks)]
        
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df['word_count'] = chunks_df['text'].apply(lambda x: len(x.split()))
        
        display(chunks_df[['chunk_number', 'word_count', 'text']])
    else:
        print("No chunks were created (text might be shorter than chunk size).")
    print("-" * 25)



def build_and_visualize_knowledge_graph(triples_df: pd.DataFrame, output_dir: str = None):
    """
    Build and visualize knowledge graph from the canonical triples using NetworkX.
    Provides multiple visualization options, graph analytics, and QA capabilities.
    """
    
    G = create_graph(triples_df)
    print("\nSaving graph in GraphML format...")
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"Saved GraphML file to: {graphml_path}")
    
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
    
    
    return G, graphml_path

   