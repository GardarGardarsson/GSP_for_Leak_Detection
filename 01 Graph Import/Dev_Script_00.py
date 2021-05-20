#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:33:38 2021

@author: gardar
"""

import torch
import fiona
import networkx as nx
import matplotlib.pyplot as plt
from s2g import ShapeGraph
from shapely.geometry import shape, LineString
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

#sg = ShapeGraph(shapefile="./graph_data/alftanes_shp/Export_Output.shp", to_graph=True)
#assert isinstance(sg.graph, nx.Graph)
#G = sg.to_networkx()
#data = from_networkx(G)


shp = './graph_data/alftanes_shp/Export_Output.shp'

with fiona.open(shp) as source:
    geoms = []
    for r in source:
        s = shape(r['geometry'])
        if isinstance(s, LineString):
            geoms.append(s)

# create ShapeGraph object from a list of lines
sg = ShapeGraph(geoms, to_graph=True)

# detect major components
mc = sg.gen_major_components()
# major components are mc[2]

# convert the largest component to networkx Graph
G = sg.to_networkx()  # equivalently sg.graph

assert isinstance(sg.graph, nx.Graph)
# %%
G = sg.to_graph()

# %%
pos = {number: coord for number,coord in G.nodes(),sg.line_info().coords}

nx.draw_networkx_nodes(G,pos,node_size=10,node_color='r')

nx.draw_networkx_edges(G,pos,edge_color='b')

plt.show()


# %%
pos = {k: v for k,v in enumerate(G.nodes())}
X=nx.Graph() #Empty graph
X.add_nodes_from(pos.keys()) #Add nodes preserving coordinates
l=[set(x) for x in G.edges()] #To speed things up in case of large objects
edg=[tuple(k for k,v in pos.items() if v in sl) for sl in l] #Map the G.edges start and endpoints onto pos
nx.draw_networkx_nodes(X,pos,node_size=100,node_color='r')
X.add_edges_from(edg)
nx.draw_networkx_edges(X,pos)
plt.xlim(450000, 470000) #This changes and is problem specific
plt.ylim(430000, 450000) #This changes and is problem specific
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('From shapefiles to NetworkX')
