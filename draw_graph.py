import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = ['train.py', 'AdversarialRunner', 'create_parallel_env', 'make_agent', 'FileWriter', 'log_stats', 'Evaluator', 'checkpoint', 'screenshot', 'pyvirtualdisplay']
G.add_nodes_from(nodes)

# Add edges to represent relationships
edges = [('train.py', 'create_parallel_env'),
         ('train.py', 'make_agent'),
         ('train.py', 'AdversarialRunner'),
         ('train.py', 'FileWriter'),
         ('train.py', 'log_stats'),
         ('train.py', 'Evaluator'),
         ('train.py', 'checkpoint'),
         ('train.py', 'screenshot'),
         ('train.py', 'pyvirtualdisplay'),
         ('AdversarialRunner', 'make_agent')]
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(15, 10))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, width=2, edge_color='gray', font_weight='bold')
plt.title("Expanded Relationship between Components", size=15)
plt.show()
