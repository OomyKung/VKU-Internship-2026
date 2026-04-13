import pickle
import networkx as nx

txt_path = "networks/facebook_combined.txt"
pkl_path = "networks/facebook_combined.pickle"

# Read edge list
G = nx.read_edgelist(txt_path, nodetype=int)

# Save as pickle
with open(pkl_path, "wb") as f:
    pickle.dump(G, f)

print(f"Saved graph to {pkl_path}")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")