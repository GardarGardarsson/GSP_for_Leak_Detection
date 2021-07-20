#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+-------------------------------+
|                               |
|   V I S U A L I S A T I O N   | 
|                               |
+-------------------------------+

Created on Tue Jul  6 09:23:33 2021

@author: gardar
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt


# Helper function for visualisation
def visualise(G, pos=None, color='blue', epoch=None, loss=None, figsize=(32,32), **kwargs):
    
    # If coordinates were not passed to the plotting function
    if not pos:
        # Plot as spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Set up canvas
    plt.figure(figsize=figsize, dpi = 300)
    plt.xticks([])
    plt.yticks([])

    # If this is a trained GNN
    if torch.is_tensor(G):
        G = G.detach().cpu().numpy()
        plt.scatter(G[:, 0], G[:, 1], s=140, c=color, cmap="Set2", **kwargs)
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
            
    # If this is a networkx graph
    else:
        #nx.draw(G, {n:[n[0], n[1]] for n in list(G.nodes)}, ax=plt.gca(), node_size=2)
        nx.draw_networkx(G, pos=pos, with_labels=False,
                         node_color=color, cmap="Set2")
        
    # Display
    plt.show()

