#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:56:31 2021

Utility library for helper functions

@author: gardar
"""

import networkx as nx
import pandas as pd

def list_of_dict_search(search_item, search_key, list_of_dicts):
    '''
    Helper function for searching within a list of dictionaries
    Returns boolean 'True' if a hit is found, 'False' otherwise
    
    Parameters
    ----------
    search_item : Item to search for
    search_key : Which key to query 
    list_of_dicts : The list of dictionaries to query
    
    Returns
    -------
    wasFound : Boolean True if hit, else False
    '''    
    
    # Initialise the hit value as false
    wasFound = False
    
    # For each dictionary within the list
    for dictionary in list_of_dicts:
        
        # Check if the value assigned to the search key matches the search item
        if dictionary[search_key] == search_item:
        
            # If so, we have a hit...
            wasFound = True
            
            # ... and can stop iteration
            break
        
        # If not
        else:
            
            # We continue searching
            continue
    
    # Return the hit status
    return wasFound

if __name__=="__main__":
        
    # List of dictionary search sanity check
    l_of_d = [{'id':1,
               'x' : -215.6,
               'y' : 513.2}
              , 
              {'id': 97,
               'x' : 21.9,
               'y' : 78.9}]
    
    item = 97
    key  = 'id'
    wasFound = list_of_dict_search(search_item = item, search_key = key, list_of_dicts=l_of_d) 
    
    # Print search result
    print("The value: '{search}' {result} found".format(search=item, 
                                                        result=('was' if wasFound else 'was NOT')))

def pipeByneighbourLookup(node1, node2, pipe_by_neighbours, verbose=False):
    '''
    A lookup function that returns a pipe's namestring given two nodes.
    The namestring is returned if the two nodes are directly connected by a pipe.
    Otherwise it returns None.
    The verbose setting may be used to surpress a notification that indicates
    that no connection exists between the two nodes.
    
    Parameters
    ----------
    int node1 : Integer value referring to the number of node 1
    int node2 : Integer value referring to the number of node 2
    dict pipe_by_neighbour : Dictionary of {'[node1,node2]' : 'pipe'}, on which lookup is performed.
    bool verbose : Print out on/off
    
    Returns
    -------
    str pipe_name : String indicating the name of the connecting pipe, None otherwise
    '''
    try:
        return pipe_by_neighbours[str([node1,node2])]   # If we don't find the first combination
    except:
        try:                                            # We try the next
            return pipe_by_neighbours[str([node2,node1])]
        except:                                         # And if we still don't find it
            if verbose:
                print('Nodes are not connected by a pipe')
            return None                                 # We return nothing

def discoverNeighbourhood(pipe, neighbours_by_pipe, pipe_by_neighbours, graph, k=3):
    '''
    A function for returning a list of nodes and pipes in the k-neighbourhood of
    a given pipe.
    
    Parameters
    ----------
    str pipe : String indicating name of pipe, e.g. 'p257'
    dict neighbours_by_pipe : Dictionary of the form {'pipe' : [node1, node2]}
    dict pipe_by_neighbours : Dictionary of the form {'[node1, node2]' : 'pipe'}
    nxGraph graph : A networkx graph object
    int k : The k-neighbourhood to be discovered (e.g. k = 3, 3-hop neighbourhood)
    
    Returns
    -------
    list pipes_in_neighbourhood : A list of pipe namestrings in the k-hop neighbourhood of the given pipe
    list n_hop_neighbours : A list of node names in the k-hop neighbourhood of the given pipe
    '''
    leaky_nodes = neighbours_by_pipe[pipe]
    
    node_1_neighbours = nx.single_source_shortest_path_length(graph, leaky_nodes[0], cutoff=k)
    node_2_neighbours = nx.single_source_shortest_path_length(graph, leaky_nodes[1], cutoff=k)
    
    n_hop_neighbours = [unique for unique in pd.DataFrame([node_1_neighbours, node_2_neighbours]).columns]

    pipes_in_neighbourhood = []                                                    # List of neighbourhood pipes
    for neighbour1 in n_hop_neighbours:                                            # For neighbour 1
        for neighbour2 in n_hop_neighbours:                                        # For neighbour 2
            pipe = pipeByneighbourLookup(neighbour1,neighbour2,pipe_by_neighbours) # Look for a connecting pipe ...
            if pipe:                                                               # ... with the neighbours and if found:
                pipes_in_neighbourhood.append(pipe)                                # .... add it to the list
    
    pipes_in_neighbourhood = list(dict(zip(pipes_in_neighbourhood,[pipes_in_neighbourhood.count(i) for i in pipes_in_neighbourhood])).keys())
    
    return pipes_in_neighbourhood, n_hop_neighbours

def determineWindowSize(window='5d',sampling_rate='5min'):
    '''
    One-liner to convert a time period to number of intervals given a sampling rate.
    
    Parameters
    ----------
    pd.DateTime window : Time period, e.g. '5h'
    pd.DateTime sampling_rate : Sampling rate, e.g. '1s'
    
    Returns
    -------
    int interval : The number of intervals in the timeperiod given the sampling rate
    '''
    return int(pd.Timedelta(window) / pd.Timedelta(sampling_rate))
