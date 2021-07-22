#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+-------------------------------+
|                               |
|    I M P O R T   M O D E L    | 
|                               |
+-------------------------------+

Created on Mon Jul  5 18:47:11 2021

@author: gardar
"""

# Import a custom tool for converting EPANET .inp files to networkx graphs
from utils.epanet_loader import get_nx_graph

# An object oriented library for handling EPANET files in Python
import epynet 

# Import the networkx library
import networkx as nx

# Matplotlib functionality
import matplotlib.pyplot as plt

# Function for visualisation
from utils.visualisation import visualise

# Torch from graph conversion tool
from torch_geometric.utils import from_networkx



'''
C O N V E R T   E P A N E T   T O   G R A P H
'''

# Set the path to the EPANET input file
pathToWDN = './BattLeDIM/L-TOWN.inp'

# Other EPANET models we might want to look at
# pathToWDN = './water_networks/anytown.inp'   
# pathToWDN = './water_networks/ctown.inp'
# pathToWDN = './water_networks/richmond.inp'

# Import the .inp file using the EPYNET library
wdn = epynet.Network(pathToWDN)

# Solve hydraulic model for a single timestep
wdn.solve()

# Convert the file using a custom function, based on:
# https://github.com/BME-SmartLab/GraphConvWat 
G , pos , head = get_nx_graph(wdn, weight_mode='pipe_length', get_head=True)

'''
V I S U A L I S E   G R A P H
'''

# Perform min-max scaling on the head data, scale it to the interval [0,1]
head  = (head - head.min()) / (head.max() - head.min()) # Try standard scaling

# Generate a colormap
cmap  = plt.get_cmap('coolwarm')

# Fit the datapoints to the colormap
color = cmap(head)

# Visualise the the model using our visualisation utility
visualise(G, pos=pos, color = color, figsize = (60,32), edge_labels=True)


#%%

'''
C O N V E R T   P H Y S I C A L   G R A P H   T O    C O M P U T A T I O N   G R A P H 
'''

# Convert networkx graph to 'torch-geometric.data.Data' object
data = from_networkx(G)