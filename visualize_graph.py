import networkx as nx
from pyvis.network import Network

# Load the GraphML file
graphml_file = "graph_chunk_entity_relation.graphml"
try:
    G = nx.read_graphml(graphml_file)
except FileNotFoundError as e:
    friendly_error_msg = f"The file {graphml_file} was not found on this path. Make sure it was generated during indexation."
    raise Exception(friendly_error_msg)

# Create a Pyvis network
net = Network(notebook=True, width="100vw", height="100vh", directed=False)

# Convert the NetworkX graph to Pyvis with node data as tooltip
for node, data in G.nodes(data=True):
    net.add_node(node, label=node, title=str(data))

# Display the weight as edge label
for source, target, data in G.edges(data=True):
    weight = data.get("weight", "")
    net.add_edge(source, target, label=str(weight))

# Show the network
net.show('knowledge_graph.html')
