import streamlit as st
import os
import tempfile
import json
import pandas as pd
import base64
from pathlib import Path
import networkx as nx
from pyvis.network import Network
from collections import Counter

from utils.pdf_splitter import extract_text_from_pdf, chunk_text_with_tables
from service.triples_generation import parallel_process_chunks, normalize_and_deduplicate_triples
from service.triple_canonicalize import canonicalize_schema, parallel_schema_processing, canonicalize_entities
from service.visulization import visualize_chunks, build_and_visualize_knowledge_graph
from utils.prompts import extraction_system_prompt
from utils.graph_builder import create_graph
from rag_service.retriever import query_graph_rag

st.set_page_config(
    page_title="Knowledge Graph Builder",
    page_icon="🕸️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .graph-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .progress-container {
        margin: 20px 0;
        padding: 15px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

def generate_html_visualization(graph_to_render: nx.Graph, output_path: str):
    """Generate an HTML visualization of the provided graph using pyvis."""
    if graph_to_render.number_of_nodes() == 0:
        print("Warning: Graph to render is empty. Cannot generate visualization.")
        with open(output_path, 'w') as f:
            f.write("<html><body>Graph is empty or filtered out.</body></html>")
        return False
    
    try:
        # Create a pyvis network
        net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", directed=graph_to_render.is_directed())
        
        # Add nodes from the subgraph
        for node in graph_to_render.nodes():
            # title = f"{node}\nDegree: {G.degree(node)}" # Example title
            net.add_node(node, label=str(node), title=str(node))
        
        # Add edges from the subgraph
        for source, target, data in graph_to_render.edges(data=True):
            relation = data.get('edge', data.get('relation', ''))
            net.add_edge(source, target, title=str(relation), label=str(relation))
        
        net.set_options("""
        {
            "nodes": {
                "font": {"size": 12, "face": "Tahoma"}
            },
            "edges": {
                "color": {"inherit": false},
                "smooth": {"type": "continuous", "forceDirection": "none"},
                "font": {"size": 10, "align": "middle"},
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
            },
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "stabilization": {"iterations": 1000}
            },
            "interaction": {"navigationButtons": true, "keyboard": true}
        }
        """)
        
        # Save the graph
        net.save_graph(output_path)
        return True
    except Exception as e:
        st.error(f"Error generating HTML visualization: {str(e)}")
        return False

def process_pdf(uploaded_file):
    """Process an uploaded PDF file and create a knowledge graph."""
    company_name = os.path.splitext(uploaded_file.name)[0].split('_')[0].upper()
    
    # Create company directory structure
    company_dir = os.path.join("companies", company_name)
    output_dir = os.path.join(company_dir, "graph_output")
    os.makedirs(company_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define path for normalized triples cache
    normalized_triples_path = os.path.join(company_dir, 'normalized_triples.json')

    normalized_triples = None
    progress = st.container()
    
    with progress:
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)

        # Check if normalized triples already exist
        if os.path.exists(normalized_triples_path):
            st.info(f"Found existing normalized triples: {normalized_triples_path}. Loading...")
            try:
                with open(normalized_triples_path, 'r') as f:
                    normalized_triples = json.load(f)
                st.success(f"✅ Loaded {len(normalized_triples)} normalized triples from cache.")
                st.info("Skipping Steps 1-4 (Extraction, Chunking, Triple Generation, Normalization)")
            except Exception as e:
                st.error(f"Error loading normalized triples from cache: {str(e)}")
                st.warning("Proceeding with full processing pipeline.")
                normalized_triples = None 
 
        if normalized_triples is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            # Step 1: Extract text
            with st.spinner('Extracting text from PDF...'):
                text = extract_text_from_pdf(pdf_path)
                st.success('✅ Text extraction complete')
            
            # Step 2: Chunk text
            with st.spinner('Chunking text...'):
                chunks = chunk_text_with_tables(text)
                st.success(f'✅ Text split into {len(chunks)} chunks')
                
               
                try:
                    chunks_path = os.path.join(company_dir, 'source_chunks.json')
                    # Store as dict {index: text} for easier lookup later
                    chunks_dict = {i: chunk_text for i, chunk_text in enumerate(chunks)} 
                    with open(chunks_path, 'w', encoding='utf-8') as f:
                        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
                    st.info(f"Saved source text chunks to {chunks_path}")
                except Exception as e:
                    st.error(f"Error saving source chunks: {str(e)}")
                
                with st.expander("View chunk details"):
                    visualize_chunks(chunks)
            
            # Step 3: Process chunks
            with st.spinner('Processing chunks...'):
                all_triples, all_failures = parallel_process_chunks(
                    chunks, 
                    extraction_system_prompt,
                    company_name,
                    batch_size=10, 
                    max_workers=4
                )
                st.success(f'✅ Extracted {len(all_triples)} triples')
                if all_failures:
                    st.warning(f'⚠️ {len(all_failures)} chunks failed processing')
            
            # Step 4: Normalize and deduplicate triples
            with st.spinner('Normalizing and deduplicating triples...'):
                normalized_triples = normalize_and_deduplicate_triples(all_triples)
                st.success(f'✅ Normalized to {len(normalized_triples)} unique triples')

                try:
                    with open(normalized_triples_path, 'w') as f:
                        json.dump(normalized_triples, f, indent=2)
                    st.info(f"Saved normalized triples to {normalized_triples_path}")
                except Exception as e:
                    st.error(f"Error saving normalized triples: {str(e)}")
            
            os.unlink(pdf_path)

        if normalized_triples is None:
             st.error("Critical error: Normalized triples could not be obtained. Aborting.")
             st.markdown('</div>', unsafe_allow_html=True)
             return None, None, None, None 


        # Step 5: Define schema
        with st.spinner('Defining schema...'):
            relation_definitions, failed_relations = parallel_schema_processing(
                normalized_triples,
                max_workers=4
            )
            st.success('✅ Schema definition complete')
            if failed_relations:
                st.warning(f'⚠️ Failed to define {len(failed_relations)} relations')
        
        # Step 6: Canonicalize schema
        with st.spinner('Canonicalizing schema...'):
            canonical_triples, canonical_relation_map = canonicalize_schema(
                normalized_triples,
                relation_definitions
            )
            st.success('✅ Schema canonicalization complete')
            
        # Step 7: Canonicalize entities
        with st.spinner('Canonicalizing entities...'):
            canonical_triples, canonical_entity_map = canonicalize_entities(
                canonical_triples,
                similarity_threshold=0.94
            )
            st.success('✅ Entity canonicalization complete')
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Save canonical triples
    with open(os.path.join(company_dir, 'canonical_triples.json'), 'w') as f:
        json.dump(canonical_triples, f, indent=2)
    
    # Save schema information
    schema_info = {
        'relation_definitions': relation_definitions,
        'relation_mapping': canonical_relation_map,
        'entity_mapping': canonical_entity_map,
        'failed_relations': failed_relations
    }
    with open(os.path.join(company_dir, 'schema_info.json'), 'w') as f:
        json.dump(schema_info, f, indent=2)
    
    # Convert triples to DataFrame format for visualization
    triples_df = pd.DataFrame(canonical_triples)
    if 'subject' in triples_df.columns and 'object' in triples_df.columns:
        triples_df = triples_df.rename(columns={'subject': 'node_1', 'object': 'node_2', 'predicate': 'edge'})
    elif 'head' in triples_df.columns and 'tail' in triples_df.columns:
        triples_df = triples_df.rename(columns={'head': 'node_1', 'tail': 'node_2', 'relation': 'edge'})
    
    # Build and visualize knowledge graph
    with st.spinner('Building knowledge graph...'):
        G, graphml_path = build_and_visualize_knowledge_graph(triples_df, output_dir)
        st.success('✅ Knowledge graph created')
        
        html_path = os.path.join(output_dir, "knowledge_graph_interactive.html")
        if not os.path.exists(html_path):
            st.info("Creating custom interactive visualization...")
            success = generate_html_visualization(G, html_path)
            if success:
                st.success("✅ Custom visualization created")
            else:
                st.warning("Failed to create custom visualization. You can still view the graph structure.")
    
    return company_name, G, canonical_triples, schema_info

def display_graph_metrics(G):
    """Display key metrics about the knowledge graph."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <h3>Nodes</h3>
                <h2>{G.number_of_nodes():,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <h3>Edges</h3>
                <h2>{G.number_of_edges():,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col3:
        n_relations = len(set(nx.get_edge_attributes(G, 'edge').values()))
        st.markdown(
            f"""<div class="metric-card">
                <h3>Unique Relations</h3>
                <h2>{n_relations:,}</h2>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col4:
        density = nx.density(G)
        st.markdown(
            f"""<div class="metric-card">
                <h3>Graph Density</h3>
                <h2>{density:.6f}</h2>
            </div>""", 
            unsafe_allow_html=True
        )

def get_available_companies():
    """Get list of companies with processed data."""
    companies_dir = "companies"
    if not os.path.exists(companies_dir):
        os.makedirs(companies_dir)
    return [d for d in os.listdir(companies_dir) 
            if os.path.isdir(os.path.join(companies_dir, d))]

@st.cache_data
def load_company_data(company_name):
    """Load processed data for a company."""
    company_dir = os.path.join("companies", company_name)
    graph_output_dir = os.path.join(company_dir, "graph_output")
    
    try:
        # Load canonical triples
        with open(os.path.join(company_dir, 'canonical_triples.json'), 'r') as f:
            canonical_triples = json.load(f)
        
        # Load schema info
        with open(os.path.join(company_dir, 'schema_info.json'), 'r') as f:
            schema_info = json.load(f)
        
        # Load graph
        graphml_path = os.path.join(graph_output_dir, 'knowledge_graph.graphml')
        if os.path.exists(graphml_path):
            #print(f"CACHE MISS/RECOMPUTE: Loading graph for {company_name} from {graphml_path}")
            G = nx.read_graphml(graphml_path)
            return G, canonical_triples, schema_info
        else:
            st.error(f"No graph found at {graphml_path}")
            return None, canonical_triples, schema_info
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def display_graph_analysis(G):
    """Display analysis of the graph structure."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Connected Entities")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        for node, degree in top_nodes:
            st.write(f"- {node}: {degree} connections")
    
    with col2:
        st.subheader("Most Common Relations")
        relations = [d.get('edge', d.get('relation', '')) for _, _, d in G.edges(data=True)]
        relation_counts = Counter(relations)
        for rel, count in relation_counts.most_common(10):
            st.write(f"- {rel}: {count} occurrences")

def main():
    st.title("🕸️ Knowledge Graph Builder")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Get available companies
    available_companies = get_available_companies()
    
    # File upload section
    st.sidebar.header("Process New Document")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file (format: TICKER_10k.pdf)", type=['pdf'])
    
    if uploaded_file:
        company_name = os.path.splitext(uploaded_file.name)[0].split('_')[0].upper()
        st.sidebar.info(f"Detected company ticker: {company_name}")
        
        if st.sidebar.button("Process PDF"):
            if company_name in available_companies:
                st.sidebar.warning(f"Company {company_name} already exists. Existing data will be overwritten.")
            
            company_name, G, triples, schema_info = process_pdf(uploaded_file)
            if company_name:
                st.success(f"Successfully processed {company_name} data!")
                st.rerun()
            else:
                st.error("Processing failed. Please check logs or try again.")
    
    # Company selection for viewing
    if available_companies:
        st.sidebar.header("View Existing Graphs")
        selected_company = st.sidebar.selectbox(
            "Choose a company to view",
            [""] + available_companies,
            format_func=lambda x: "Select a company..." if x == "" else x
        )
        
        if selected_company:
            # Load data (cached)
            G, triples, schema_info = load_company_data(selected_company)
            
            if G is not None and triples is not None:
                # Display metrics
                display_graph_metrics(G)
                
                # Create tabs
                tab_titles = ["Graph Visualization", "Triple Explorer", "Schema Info", "Query Graph (RAG)"]
                tabs = st.tabs(tab_titles)
                
                # Tab 1: Graph Visualization (Modified for Subgraph)
                with tabs[0]:
                    st.header("Knowledge Graph Visualization")
                    
                    if G.number_of_nodes() == 0:
                        st.warning("Graph is empty. Cannot display visualization.")
                    else:
                        # --- Add Degree Slider --- 
                        max_degree = 0
                        if G.number_of_nodes() > 0:
                            degrees = [d for n, d in G.degree()]
                            if degrees:
                                max_degree = max(degrees)
                        
                        # Set slider range dynamically, but cap max reasonablely
                        slider_max = min(max_degree, 50) 
                        min_degree = st.slider("Filter nodes by minimum degree:", min_value=0, max_value=slider_max, value=1, step=1)
                        
                        # --- Filter Graph --- 
                        nodes_to_keep = [n for n, d in G.degree() if d >= min_degree]
                        subgraph = G.subgraph(nodes_to_keep)
                        
                        st.info(f"Displaying subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges (min degree = {min_degree}).")

                        # --- Generate and Display HTML for Subgraph --- 
                        # Use a temporary file for the visualization
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                            html_path = tmp_file.name
                        
                        success = generate_html_visualization(subgraph, html_path) # Pass subgraph here
                        
                        if success and os.path.exists(html_path):
                            try:
                                with open(html_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=800, scrolling=True)
                            except Exception as e:
                                st.error(f"Error reading visualization file: {e}")
                            finally:
                                # Clean up temporary file
                                try:
                                    os.unlink(html_path)
                                except OSError as e:
                                    print(f"Error deleting temp file {html_path}: {e}")
                        elif not success:
                            st.error("Failed to generate visualization for the subgraph.")
                        else: 
                            st.error("Visualization file was not created.")
                            if os.path.exists(html_path):
                                os.unlink(html_path)

                    # --- Graph analysis section (Operates on Full Graph G) --- 
                    with st.expander("Graph Analysis (Full Graph)"):
                        display_graph_analysis(G) # Analyze the original full graph
                
                # Tab 2: Triple Explorer
                with tabs[1]:
                    st.header("Triple Explorer")
                    
                    # Search functionality
                    search_term = st.text_input("Search triples", "")
                    
                    # Filter triples based on search
                    filtered_triples = triples
                    if search_term:
                        filtered_triples = [t for t in triples 
                                          if (search_term.lower() in t.get('subject', '').lower() 
                                              or search_term.lower() in t.get('predicate', '').lower() 
                                              or search_term.lower() in t.get('object', '').lower())]
                    
                    # Display as dataframe
                    triples_df = pd.DataFrame(filtered_triples)
                    st.dataframe(triples_df, use_container_width=True)
                    
                    # Export functionality
                    if st.button("Export to CSV"):
                        csv = triples_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{selected_company}_triples.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                # Tab 3: Schema Info
                with tabs[2]:
                    st.header("Schema Information")
                    
                    # Relation definitions
                    st.subheader("Relation Definitions")
                    if schema_info and 'relation_definitions' in schema_info:
                        relations_df = pd.DataFrame([
                            {"Relation": k, "Definition": v} 
                            for k, v in schema_info['relation_definitions'].items()
                        ])
                        st.dataframe(relations_df, use_container_width=True)
                    
                    # Canonical mappings
                    st.subheader("Canonical Mappings")
                    if schema_info and 'relation_mapping' in schema_info:
                        mappings_df = pd.DataFrame([
                            {"Original": k, "Canonical": v} 
                            for k, v in schema_info['relation_mapping'].items()
                        ])
                        st.dataframe(mappings_df, use_container_width=True)
                    
                    # Failed relations
                    if schema_info and 'failed_relations' in schema_info and schema_info['failed_relations']:
                        st.subheader("Failed Relations")
                        st.warning(f"The following {len(schema_info['failed_relations'])} relations failed to process:")
                        st.write(schema_info['failed_relations'])

                # Tab 4: Query Graph (RAG)
                with tabs[3]:
                    st.header(f"Query {selected_company} Knowledge Graph (RAG)")
                    
                    user_question = st.text_area(f"Ask a question about {selected_company}:", key=f"rag_question_{selected_company}")
                    query_button = st.button("Ask", key=f"rag_button_{selected_company}")
                    
                    if query_button and user_question:
                        with st.spinner("Querying Knowledge Graph..."):
                            # Update the call to receive both answer and context
                            answer, context_used = query_graph_rag(user_question, selected_company)
                        
                        # Display the answer
                        st.markdown("**Answer:**")
                        st.markdown(answer)
                        
                        # Display the context used, if any meaningful context exists
                        if context_used:
                            if context_used.startswith("Entities found in graph:"):
                                 with st.expander("Graph Entities Found (No Context Triples)"):
                                     st.info(context_used) # Display the info message
                            else:
                                with st.expander("View Graph Context Used"):
                                    st.text(context_used) 
                                    
                    elif query_button and not user_question:
                        st.warning("Please enter a question.")

            else:
                st.error(f"Failed to load data for {selected_company}. Try processing the document again or check file paths.")
    else:
        st.info("No processed documents available. Please upload a PDF to begin.")

if __name__ == "__main__":
    main() 