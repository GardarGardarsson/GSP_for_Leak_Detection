#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:42:25 2021

@author: gardar
"""

import fiona
import networkx as nx
from shapely.geometry import shape

# Open geometrical features of shapefile
geoms =[shape(feature['geometry']) for feature in fiona.open("./graph_data/alftanes_shp/Export_Output.shp")]

# Construct a Graph from the features
G = nx.Graph()

# Add line features as edges to graph
for line in geoms:
    for seg_start, seg_end in zip(list(line.coords),list(line.coords)[1:]):
        G.add_edge(seg_start, seg_end) 
        
#%% 