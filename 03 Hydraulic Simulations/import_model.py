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

# Function for visualisation
from utils.visualisation import visualise

# Torch from graph conversion tool
from torch_geometric.utils import from_networkx


'''
C O N V E R T   E P A N E T   T O   G R A P H
'''

# Set the path to the EPANET input file
pathToWDN = './BattLeDIM/L-TOWN.inp'

# Import the .inp file using the EPYNET library
wdn = epynet.Network(pathToWDN)

# Solve hydraulic model for a single timestep
# wdn.solve()

# Convert the file using a custom function, based on:
# https://github.com/BME-SmartLab/GraphConvWat 
G , pos = get_nx_graph(wdn, weight_mode='inv_pipe_length')

'''
V I S U A L I S E   G R A P H
'''

# Convert networkx graph to 'torch-geometric.data.Data' object
data = from_networkx(G)

'''
Finish up!
'''
titles = {'title' : 'A Graph'}

# Visualise the the model using our visualisation utility
visualise(G, pos=pos, color = data.y, figsize = (20,16), **titles)
