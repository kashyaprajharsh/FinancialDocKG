import networkx as nx
import matplotlib.pyplot as plt

def create_graph(df):
    G = nx.from_pandas_edgelist(df, 'node_1', 'node_2', edge_attr='edge', create_using=nx.MultiGraph())
    nx.draw(G, with_labels=True)
    return G

def query_sub_graph(G, query_node):
    neighbors = list(G.neighbors(query_node)) + [query_node]
    subgraph = G.subgraph(neighbors)

    pos = nx.spring_layout(subgraph)

    plt.figure(figsize=(8, 8))

    node_size = 2000
    node_color = 'lightblue'
    font_color = 'black'
    font_weight = 'bold'
    font_size = 8
    edge_color = 'gray'
    edge_style = 'dashed'

    nx.draw(subgraph, pos, with_labels=True, node_size=node_size, node_color=node_color, font_color=font_color, font_size=font_size,
            font_weight=font_weight, edge_color=edge_color, style=edge_style)

    plt.title(f"Graph of Node: {query_node}")

    plt.savefig('subgraph.png')
    #plt.show()