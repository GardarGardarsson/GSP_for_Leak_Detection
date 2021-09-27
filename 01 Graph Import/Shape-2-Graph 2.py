#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:22:34 2021

Veitur have supplied shapefiles for the district metered area (DMA) of Álftanes

`./graph_data/alftanes_shp/Export_Output.shp`

Shapefiles are a data structure that stores the coordinates of line objects.
We shall now see if this data can be turned into a graph of the `networkx` library.

We will first be reading the `.shp` files into a `geopandas` dataframe. 
From that dataframe, we'll try constructing the `networkx` graph using the `momepy` package 

@author: gardar
"""

import momepy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

'''
We now read the shapefile for the district metered area into a dataframe
Water distribution network, abbr. `wdn`.
'''
wdn = gpd.read_file('./graph_data/alftanes_shp/Export_Output.shp')

'''
The coordinate system can be set to `EPSG 3857`, which is widely supported by 
openly accessible map tile providers
'''
wdn = wdn.to_crs(epsg=3857)


'''
We have to convert this LineString geopandas dataframe (geo-dataframe, `gdf`) to `networkx.Graph`.<br> 
We use `momepy.gdf_to_nx` to convert geo-dataframe to `networkx`graph.<br>
To convert a graph back to a geo-dataframe one uses `momepy.nx_to_gdf`. 
The former, `gdf_to_nx`, supports both primal and dual graphs. <br>

- **Primal** approach will save length of each segment to be used as a weight later
- **Dual** will save the angle between segments (allowing angular centrality).

Let's go with the *primal* approach for now and see how it goes
'''
graph = momepy.gdf_to_nx(wdn, approach='primal')


'''
1. We'll plot the base geometry from the shapefile contained in the geo-dataframe
2. The graph constructed from the geo-dataframe
3. An overlay of the graph on the actual WDN
'''
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, dpi = 300)

for i, subplot in enumerate(ax):
    subplot.set_title(("Alftanes WDN", "Primal graph recreation", "Overlay")[i])
    subplot.axis("off")

# 1. Base network from geo-dataframe:
wdn.plot(color='#e15d46', ax=ax[0])

# 2. Graph figure    
nx.draw(graph, {n:[n[0], n[1]] for n in list(graph.nodes)}, ax=ax[1], node_size=2)

# 3. Graph - base overlay
wdn.plot(color='#e15d46', ax=ax[2], zorder=-1)
nx.draw(graph, {n:[n[0], n[1]] for n in list(graph.nodes)}, ax=ax[2], node_size=2)

print(list(graph.nodes))