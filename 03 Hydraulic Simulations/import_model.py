#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

+-------------------------------+
|                               |
|    I M P O R T   M O D E L    | 
|                               |
+-------------------------------+

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

# PyTorch deep learning framework
import torch

# Import the networkx library
import networkx as nx

# Import Pandas for data handling
import pandas as pd

# Import numpy for array handling
import numpy as np

# Matplotlib for generating graphics
import matplotlib.pyplot as plt

# PyTorch from graph conversion tool
from torch_geometric.utils import from_networkx

# Train-test split with shuffle 
from sklearn.model_selection import train_test_split

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

# Function for visualisation
from utils.visualisation import visualise

# EPANET simulator, used to generate nodal pressures from the nominal model
from utils.epanet_simulator import epanetSimulator

# SCADA timeseries dataloader
from utils.data_loader import battledimLoader, dataCleaner, dataGenerator

# PyTorch early stopping callback
from utils.early_stopping import EarlyStopping

# Metrics
from utils.metrics import Metrics

# GNN model library
from modules.gnn_models import ChebNet


#%% Parse arguments

# Main loop
if __name__ == "__main__" :
    
    '''
    1.   C O N F I G U R E   E X E C U T I O N   -   A R G P A R S E R
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nRunning: \t 'import_model.py' ")
    print("Using device: \t {}".format(device))
    
    # Generate an argument parser
    parser = argparse.ArgumentParser()
    
    # Configure the argparser:
    # Choice of water-distribution network to work with
    parser.add_argument('--wdn',
                        default = 'ltown',
                        choices = ['ltown','anytown','ctown','richmond'],
                        type    = str,
                        help    = "Choose WDN: ['ltown', 'anytown', 'ctown', 'richmond']")
    
    # Choice of visualising the created graph
    parser.add_argument('--visualise',
                        default = 'no',
                        choices = ['yes','no'],
                        type    = str,
                        help    = "Visualise graph after import? 'yes' / 'no' ")
    
    # Choice of visualising the created graph
    parser.add_argument('--visualiseWhat',
                        default = 'sensor_location',
                        choices = ['sensor_location','pressure_signals'],
                        type    = str,
                        help    = "What to visualise: ['sensor_locations','pressure_signals']")
    
    # Choice of rescaling the timeseries data
    parser.add_argument('--scaling',
                        default = 'standard',
                        choices = ['standard','minmax',None],
                        type    = str,
                        help    = "Rescale timeseries data: ['standard','minmax']")
    
    # Push the passed arguments to the 'args' variable
    args = parser.parse_args()
    
    # Printout of configuration for the following execution
    print("\nArguments for session: \n" + 22*"-" + "\n{}".format(args) + "\n")
    
    
    #%%n Set the filepaths for the execution
    
    '''
    2.   C O N F I G U R E   E X E C U T I O N   -   P A T H S
    '''
    
    print('Setting environment paths...\n')
    
    # Set the path to the EPANET input file
    if args.wdn == 'ltown':
        path_to_wdn = './BattLeDIM/L-TOWN.inp' # Do I need to distinguish between REAL and NOMINAL EPANET inps here?        
    elif args.wdn == 'anytown':
        path_to_wdn = './water_networks/anytown.inp'   
    elif args.wdn == 'ctown':
        path_to_wdn = './water_networks/ctown.inp'
    elif args.wdn == 'richmond':
        path_to_wdn = './water_networks/richmond.inp'
    else:
        print('Unknown WDN, exiting')
        exit()
    
    #%% Convert EPANET hydraulic model to networkx graph
    
    '''
    3.   C O N V E R T   E P A N E T   T O   G R A P H
    '''
    
    print('Importing EPANET file and converting to graph...\n')
    
    # Import the .inp file using the EPYNET library
    wdn = epynet.Network(path_to_wdn)
    
    # Solve hydraulic model for a single timestep
    wdn.solve()
    
    # Convert the file using a custom function, based on:
    # https://github.com/BME-SmartLab/GraphConvWat 
    G , pos , head = get_nx_graph(wdn, weight_mode='inv_pipe_length', get_head=True)
    
    
    #%% Read in dataset configuration (.yml)
    
    '''
    4.   I M P O R T   D A T A S E T   C O N F I G
    '''
    
    print('Importing dataset configuration...\n')
    
    # Open the dataset configuration file
    with open('./BattLeDIM/dataset_configuration.yml') as file:
        
        # Load the configuration to a dictionary
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Generate a list of integers, indicating the number of the node
    # at which a  pressure sensor is present
    sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]
    
    
    #%% Visualise the created graph
    
    '''
    5.   V I S U A L I S E   G R A P H   &   P R I N T   I N F O 
    '''
    
    # If user chose to generate plot
    if args.visualise == 'yes':
       
        print('Generating plot of graph...\n') 
       
        # Generate rendering based on configuration
        
        # Plot pressure signals
        if args.visualiseWhat == 'pressure_signals':
            
            # Perform min-max scaling on the head data, scale it to the interval [0,1]
            colormap  = (head - head.min()) / (head.max() - head.min()) # Try standard scaling
        
        # Plot the location of the pressure sensors
        elif args.visualiseWhat == 'sensor_location':
            colormap = pd.Series([1.0 if i in sensors else 0.0 for i in range(1,783)])
        
        # Generate a colormap
        cmap  = plt.get_cmap('coolwarm')
        
        # Fit the datapoints to the colormap
        color = cmap(colormap)
        
        # Visualise the the model using our visualisation utility
        axis = visualise(G, pos=pos, color = color, figsize = (60,32), edge_labels=True)
        
        plt.show()

    # We may want to find the largest subgraph diameter
    largest_subgraph_diameter = 0
    
    # For each component of the imported network
    for c in nx.connected_components(G):
        graph = G.subgraph(c)                       # Generate a subgraph
        diameter = nx.diameter(graph)               # Measure its diameter
        if diameter > largest_subgraph_diameter:    # If it's the longest encountered
            diameter = largest_subgraph_diameter    # Store it  
        
    #%% Import timeseries data
    
    '''
    6.   I M P O R T   S C A D A   D A T A 
    '''
    
    print('Importing SCADA dataset...\n') 
    
    # Load the data into a numpy array with format matching the GraphConvWat problem
    pressure_2018 = battledimLoader(observed_nodes = sensors,
                                    n_nodes        = 782,
                                    path           = './BattLeDIM/',
                                    file           = '2018_SCADA_Pressures.csv')
    
    # Print information and instructions about the imported data
    msg = "The imported sensor data has shape (i,n,d): {}".format(pressure_2018.shape)
    
    print(msg + "\n" + len(msg)*"-" + "\n")
    print("Where: ")
    print("'i' is the number of observations: {}".format(pressure_2018.shape[0]))
    print("'n' is the number of nodes: {}".format(pressure_2018.shape[1]))
    print("'d' is a {}-dimensional vector consisting of the pressure value and a mask ".format(pressure_2018.shape[2]))
    print("The mask is set to '1' on observed nodes and '0' otherwise\n")
    
    print("\n" + len(msg)*"-" + "\n")
    
    #%% Generate the nominal pressure data from an EPANET simulation
    
    '''
    7.   G E N E R A T E   N O M I N A L   D A T A
    '''
    
    print('Running EPANET simulation to generate nominal pressure data...\n') 
    
    # We run an EPANET simulation using the WNTR library and the EPANET
    # nominal model supplied with the BattLeDIM competition. 
    # With this simulation, we have a complete pressure signal for all
    # nodes in the network, on which the GNN algorithm is to be trained.
    
    # Instantiate the nominal WDN model
    nominal_wdn_model = epanetSimulator(path_to_wdn)

    # Run a simulation
    nominal_wdn_model.simulate()
    
    # Retrieve the nodal pressures
    nominal_pressure = nominal_wdn_model.get_simulated_pressure()
    
    #%% Load the nominal pressure data and prepare for training
    
    '''
    8.   P R E P A R E   T R A I N I N G   D A T A    
    '''
        
    # Populate feature vector x and label vector y from the nominal pressures
    x,y = dataCleaner(pressure_df    = nominal_pressure, # Pass the nodal pressures
                      observed_nodes = sensors,          # Indicate which nodes have sensors
                      rescale        = args.scaling)     # Perform scaling on the timeseries data
    
    # Split the data into training and validation sets
    x_trn, x_val, y_trn, y_val = train_test_split(x, y, 
                                                  test_size    = 0.2,
                                                  random_state = 1,
                                                  shuffle      = False)
     
        
    #%% Convert the graph to a computation graph
    
    '''
    9. C O N V E R T   P H Y S I C A L   G R A P H   T O    C O M P U T A T I O N   G R A P H 
    '''
    def train_one_epoch(trn_gnrtr):
        model.train()
        total_loss  = 0
        for batch in trn_gnrtr:
            batch   = batch.to(device)
            optimizer.zero_grad()
            out     = model(batch)
            loss    = torch.nn.functional.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * batch.num_graphs
        return total_loss / len(trn_gnrtr.dataset)
    
    def eval_metrics(generator):
        model.eval()
        n   = len(generator.dataset)
        tot_loss        = 0
        tot_rel_err     = 0
        tot_rel_err_obs = 0
        tot_rel_err_hid = 0
        for batch in generator:
            batch   = batch.to(device)
            out     = model(batch)
            loss    = torch.nn.functional.mse_loss(out, batch.y)
            rel_err = metrics.rel_err(out, batch.y)
            rel_err_obs = metrics.rel_err(
                out,
                batch.y,
                batch.x[:, -1].type(torch.bool)
                )
            rel_err_hid = metrics.rel_err(
                out,
                batch.y,
                ~batch.x[:, -1].type(torch.bool)
                )
            tot_loss        += loss.item() * batch.num_graphs
            tot_rel_err     += rel_err.item() * batch.num_graphs
            tot_rel_err_obs += rel_err_obs.item() * batch.num_graphs
            tot_rel_err_hid += rel_err_hid.item() * batch.num_graphs
        loss        = tot_loss / n
        rel_err     = tot_rel_err / n
        rel_err_obs = tot_rel_err_obs / n
        rel_err_hid = tot_rel_err_hid / n
        return loss, rel_err, rel_err_obs, rel_err_hid
    
    # ----------------
    # Hyper-parameters
    # ----------------
    batch_size    = 40
    learning_rate = 3e-4
    decay         = 6e-6
    shuffle       = False
    epochs        = 1
    
    # Instantiate the data generators
    trn_gnrtr = dataGenerator(G, x_trn, y_trn, batch_size, shuffle)
    val_gnrtr = dataGenerator(G, x_val, y_val, batch_size, shuffle)
    
    
    model = ChebNet(np.shape(x_trn)[-1], np.shape(y_trn)[-1]).to(device)
    
    optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=args.decay),
                                  dict(params=model.conv2.parameters(), weight_decay=args.decay),
                                  dict(params=model.conv3.parameters(), weight_decay=args.decay),
                                  dict(params=model.conv4.parameters(), weight_decay=0)],
                                  lr  = learning_rate,
                                  eps = 1e-7)
    estop   = EarlyStopping(min_delta=.00001, patience=30)
    
    if args.scaling == 'standard':
        scale_y = np.std(y_trn)
        bias_y  = np.mean(y_trn)
        
    metrics = Metrics(bias_y, scale_y, device)
    best_vld_loss   = np.inf
    results = pd.DataFrame(columns=[
    'trn_loss', 'vld_loss', 'vld_rel_err', 'vld_rel_err_o', 'vld_rel_err_h'
    ])
    header  = ''.join(['{:^15}'.format(colname) for colname in results.columns])
    header  = '{:^5}'.format('epoch') + header
    for epoch in range(0, args.epoch):
        trn_loss    = train_one_epoch()
        vld_loss, vld_rel_err, vld_rel_err_obs, vld_rel_err_hid = eval_metrics(val_gnrtr)
        new_results = pd.Series({
            'trn_loss'      : trn_loss,
            'vld_loss'      : vld_loss,
            'vld_rel_err'   : vld_rel_err,
            'vld_rel_err_o' : vld_rel_err_obs,
            'vld_rel_err_h' : vld_rel_err_hid
            })
        results = results.append(new_results, ignore_index=True)
        if epoch % 20 == 0:
            print(header)
        values  = ''.join(['{:^15.6f}'.format(value) for value in new_results.values])
        print('{:^5}'.format(epoch) + values)
        if vld_loss < best_vld_loss:
            best_vld_loss   = vld_loss
            torch.save(model.state_dict(), './models')
        if estop.step(torch.tensor(vld_loss)):
            print('Early stopping...')
            break

    