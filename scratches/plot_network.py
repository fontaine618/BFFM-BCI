import torch
import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")


channel_names = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

component = torch.randn(16, 1)
network = component @ component.T
edgelist = torch.tril_indices(16, 16, offset=-1)
weights = network[edgelist[0], edgelist[1]]
G = nx.from_edgelist(edgelist.T.numpy(), create_using=nx.DiGraph)
for i, (u, v) in enumerate(G.edges()):
    G.edges[u, v]['weight'] = weights[i].item()
G = nx.relabel_nodes(G, {i: channel_names[i] for i in range(16)})
G.edges(data=True)


pos = {
    'F3': (1, 4),
    'Fz': (2, 4),
    'F4': (3, 4),
    'T7': (0, 3),
    'C3': (1, 3),
    'Cz': (2, 3),
    'C4': (3, 3),
    'T8': (4, 3),
    'CP3': (1, 2),
    'CP4': (3, 2),
    'P3': (1, 1),
    'Pz': (2, 1),
    'P4': (3, 1),
    'PO7': (1, 0),
    'Oz': (2, 0),
    'PO8': (3, 0),
}

plt.cla()
nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color='k', edgecolors='w', linewidths=2)
nx.draw_networkx_edges(G, pos=pos, width=weights.abs().numpy(), edge_color=["b" if w>0 else "r" for w in weights],
                       connectionstyle='arc3, rad=0.1', arrowsize=20, arrowstyle='-')
nx.draw_networkx_labels(G, pos=pos, font_size=12, font_color='w', font_weight='bold')
plt.axis('off')
plt.title("Network")
plt.tight_layout()
plt.show()