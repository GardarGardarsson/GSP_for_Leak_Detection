#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

+-------------------------------+
|                               |
|   E P A N E T   L O A D  E R  | 
|                               |
+-------------------------------+

Created on Mon Jul  5 18:29:39 2021

@author: gardar
"""

import numpy as np
import networkx as nx

""" 
get_nx_graph(wds, mode)

This is a function from the signal nodal pressure reconstruction project from:

    https://github.com/BME-SmartLab/GraphConvWat

The function is used for reading EPANET .inp files using the python library
EPYNET, and converting them into networkx graphs.

-------------------------------------------------------------------------------
Revision history:
-------------------------------------------------------------------------------
The function has been adjusted for this leakage detection study.
Renamed 'mode' to a more descriptive 'weight_mode'

Renamed the graph weight modes:
    binary      -> unweighted
    weighted    -> hydraulic_loss
    logarithmic -> log_hydraulic_loss
    pruned      =  pruned

... and added a new weight mode that uses just the pipe length:
    *           -> pipe_length
    
- Garðar Örn Garðarsson, 5 July 2021
  Vað, Skriðdal 

"""

def get_nx_graph(wds, weight_mode='unweighted'):
    
    # Instantiate a list of junctions
    junc_list = []
    
    # Populate the junction list
    for junction in wds.junctions:
        junc_list.append(junction.index)
        
    # Instantiate an empty graph
    G = nx.Graph()
    
    # Populate the graph
    
    # Binary mode generates an unweighted graph
    if weight_mode == 'unweighted':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=1.)
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1.)
    
    # Generate a weighted graph based on hydraulic loss calculations
    elif weight_mode == 'hydraulic_loss':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = ((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281)
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=weight)
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1.)

    # A logarithmically weighted graph
    elif weight_mode == 'log_hydraulic_loss':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = np.log10(((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281))
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1.)
    
    # Pruned?
    elif weight_mode == 'pruned':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=0.)
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=0.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=0.)

    # A pipe length weighted graph
    elif weight_mode == 'pipe_length':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = pipe.length
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1.)
      
    # An inverted pipe length weighted graph
    elif weight_mode == 'inv_pipe_length':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = (pipe.length)**(-1)
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight))
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1.)
                   
    # Return the graph object
    return G
