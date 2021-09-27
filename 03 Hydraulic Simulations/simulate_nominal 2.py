#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

+-------------------------------------------------+
|                                                 |
|    S I G N A L   R E C O N S T R U C T I O N    | 
|                                                 |
+-------------------------------------------------+

Description

Created on Mon Jul  5 18:47:11 2021

@author: gardar
'''

# --------------------------
# Importing public libraries
# --------------------------

# Operating system specific functions
import os

# Argument parser, for configuring the program execution
import argparse

# An object oriented library for handling EPANET files in Python
import epynet 

# yaml / yml configuration file support
import yaml

# Import the networkx library
import networkx as nx

# Import Pandas for data handling
import pandas as pd

# Import numpy for array handling
import numpy as np


# --------------------------
# Importing custom libraries
# --------------------------

# To make sure we don't raise an error on importing project specific 
# libraries, we retrieve the path of the program file ...
filepath = os.path.dirname(os.path.realpath(__file__))

# ... and set that as our working directory
os.chdir(filepath)

# Import a custom tool for converting EPANET .inp files to networkx graphs
from utils.epanet_loader import get_nx_graph

# EPANET simulator, used to generate nodal pressures from the nominal model
from utils.epanet_simulator import epanetSimulator


#%% Parse arguments

# Main loop
if __name__ == "__main__" :
    
    '''
    1.   C O N F I G U R E   E X E C U T I O N   -   A R G P A R S E R
    '''
    print("\nRunning: \t 'simulate_nominal.py' ")
    

    wdn_name = 'l-town'
    
    
    #%% Set the filepaths for the execution
    
    '''
    2.   C O N F I G U R E   E X E C U T I O N   -   P A T H S
    '''
    
    print('Setting environment paths...\n')
    
    # ---------------
    # Configure paths
    # ---------------
    
    path_to_data   = './data/' + wdn_name + '-data/'       # Datasets are stored here
    path_to_wdn    = './data/' + wdn_name.upper() + '.inp' # EPANET input file

    #%% Generate the nominal pressure data from an EPANET simulation
    
    '''
    6.   G E N E R A T E   N O M I N A L   D A T A   F O R   T R A I N I N G
    '''
    
        
    print('Running EPANET simulation to generate nominal pressure data...\n')
    
    # We run an EPANET simulation using the WNTR library and the EPANET
    # nominal model supplied with the BattLeDIM competition.
    # With this simulation, we have a complete pressure signal for all
    # nodes in the network, on which the GNN algorithm is to be trained.
    
    # Instantiate the nominal WDN model
    nominal_wdn_model = epanetSimulator(path_to_wdn, path_to_data)

    # Run a simulation
    nominal_wdn_model.simulate()
    
    # Retrieve the nodal pressures
    nominal_pressure = nominal_wdn_model.get_simulated_pressure()
    
    # Save the nominal pressure data frame for later imports
    nominal_pressure.to_csv(path_to_data+'nominal_pressure.csv')
    
