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

# Time for stopwatch functionality during training
import time

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
from utils.data_loader import battledimLoader, dataCleaner, dataGenerator, embedSignalOnGraph, rescaleSignal

# PyTorch early stopping callback
from utils.early_stopping import EarlyStopping

# GNN model library
from modules.torch_gnn import ChebNet

# Import user interface tools (menus etc.)
import utils.user_interface as ui

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
                        default = 'anytown',
                        choices = ['l-town','anytown','ctown','richmond'],
                        type    = str,
                        help    = "Choose WDN: ['l-town', 'anytown', 'ctown', 'richmond']")
    
    # Choice of GNN model
    parser.add_argument('--gnn',
                        default = 'chebnet',
                        choices = ['chebnet','dcrnn','gcn'],
                        type    = str,
                        help    = "Choose GNN: ['chebnet','dcrnn','gcn']")
    
    # Choice of visualising the created graph
    parser.add_argument('--visualise',
                        default = 'yes',
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
                        default = 'minmax',
                        choices = ['standard','minmax',None],
                        type    = str,
                        help    = "Rescale timeseries data: ['standard','minmax']")
                        
    # Add a custom descriptive tag
    parser.add_argument('--tag',
                        default = 'tag',
                        type    = str,
                        help    = "Add a custom, descriptive tag")
    
    # How many epochs to train the model
    parser.add_argument('--epochs',
                        default = 1, 
                        type    = int,
                        help    = "How many no. of epochs to train")
    
    # Push the passed arguments to the 'args' variable
    args = parser.parse_args()
    
    # Printout of configuration for the following execution
    print("\nArguments for session: \n" + 22*"-" + "\n{}".format(args) + "\n")
    
    
    #%% Set the filepaths for the execution
    
    '''
    2.   C O N F I G U R E   E X E C U T I O N   -   P A T H S
    '''
    
    print('Setting environment paths...\n')
    
    # ---------------
    # Configure paths
    # ---------------
    
    path_to_data   = './data/' + args.wdn + '-data/'       # Datasets are stored here
    path_to_wdn    = './data/' + args.wdn.upper() + '.inp' # EPANET input file
    path_to_logs   = './studies/logs/'                     # Log directory
    path_to_figs   = './studies/figs/'                     # Figure directory
    path_to_models = './studies/models/'                   # Saved models directory
    
    execution_no   = 1                                                                  # Initialise execution ID number
    execution_id   = args.wdn + '-' + args.gnn + '-' + args.tag + '-'                   # Initialise execution ID name
    logs           = [log for log in os.listdir(path_to_logs) if log.endswith('.csv')]  # Load all logs in directory to list
    
    while execution_id + str(execution_no) + '.csv' in logs:    # For every matching file name in the directory
        execution_no += 1                                       # Increment the execution id number
    
    execution_id   = execution_id + str(execution_no)           # Update the execution ID
    model_path     = path_to_models + execution_id + '.pt'      # Generate complete model path w/ filename
    log_path       = path_to_logs   + execution_id + '.csv'     # Generate complete log path w/ filename
    
    # If we have already trained a similar model we may wish to load its weights
    # So, we must also generate a path to that model's state dictionary
    if execution_no > 1:
        last_id         = args.wdn + '-' + args.gnn + '-' + args.tag + '-' # Initialise previous version execution ID name
        last_id         = last_id + str(execution_no - 1)                  # Execution ID number is the current number - 1  
        last_model_path = path_to_models + last_id + '.pt'                 # Generate complete path to the previously trained model
        last_log_path   = path_to_logs   + last_id + '.csv'                # Generate complete path to the previous 
        
    # -------------------
    # Adjust figure sizes
    # -------------------
    
    if args.wdn == 'anytown':   # Anytown is a very small network ...
        figsize     = (16,16)   # ... and thus requires a small frame
    else:                       # The rest however require a hi-res...
       figsize     = (60,32)    # ... if they are to be readible
    
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
    
    if args.wdn == 'l-town':
        
        print('Importing dataset configuration...\n')
        
        # Open the dataset configuration file
        with open(path_to_data + 'dataset_configuration.yml') as file:
            
            # Load the configuration to a dictionary
            config = yaml.load(file, Loader=yaml.FullLoader) 
        
        # Generate a list of integers, indicating the number of the node
        # at which a  pressure sensor is present
        sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]
        
    # -------------
    # Revise this !
    # -------------
    
    if args.wdn == 'anytown':        
        
        print('Loading timeseries sensor data...\n')
        
        # Load the training timeseries data 
        # This was generated by running the 'generate_dta.py' script for
        # the 'anytown.inp' of the GraphConvWat project of G. Hajgató et al.
        x_trn = np.load(path_to_data + 'trn_x.npy')
        y_trn = np.load(path_to_data + 'trn_y.npy')
        x_val = np.load(path_to_data + 'vld_x.npy')
        y_val = np.load(path_to_data + 'vld_y.npy')
        x_tst = np.load(path_to_data + 'tst_x.npy')
        y_tst = np.load(path_to_data + 'tst_y.npy')
        
        # Note, it's been scaled so we need the scale and bias to transform
        # it back to the original signal
        scale = np.load(path_to_data + 'scale.npy')
        bias  = np.load(path_to_data + 'bias.npy')
        
        # Get the sensor positions
        sensors = np.where(x_trn[0,:,1]>0)[0]
        
        # First node is enumerated 1 not 0 so:
        sensors += np.ones(shape=sensors.shape, dtype=int).tolist()
        
        
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
            
            # Set pressure sensors as 1 and unobserved nodes as 0
            colormap = pd.Series([1.0 if i in sensors else 0.0 for i in range(1,G.number_of_nodes()+1)])
        
        # Generate a colormap
        cmap  = plt.get_cmap('hot')
        
        # Fit the datapoints to the colormap
        color = cmap(colormap)
        
        # Visualise the the model using our visualisation utility
        axis = visualise(G, pos=pos, color = color, figsize = figsize, edge_labels=True)
        
        plt.show()
        plt.savefig(path_to_figs + execution_id + '.png')
    
    print('Calculating graph diameter...\n')
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
    
    if args.wdn == 'l-town':
        
        print('Importing SCADA dataset...\n') 
        
        # Load the data into a numpy array with format matching the GraphConvWat problem
        pressure_2018 = battledimLoader(observed_nodes = sensors,
                                        n_nodes        = 782,
                                        path           = path_to_data,
                                        file           = '2018_SCADA_Pressures.csv')
        
        # Print information and instructions about the imported data
        msg = "The imported sensor data has shape (i,n,d): {}".format(pressure_2018.shape)
        
        print(msg + "\n" + len(msg)*"-" + "\n")
        print("Where: ")
        print("'i' is the number of observations: {}".format(pressure_2018.shape[0]))
        print("'n' is the number of nodes: {}".format(pressure_2018.shape[1]))
        print("'d' is a {}-dimensional vector consisting of the pressure value and a mask ".format(pressure_2018.shape[2]))
        print("The mask is set to '1' on observed nodes and '0' otherwise")
        
        print("\n" + len(msg)*"-" + "\n")
    
    #%% Generate the nominal pressure data from an EPANET simulation
    
    '''
    7.   G E N E R A T E   N O M I N A L   D A T A
    '''
    
    if args.wdn =='l-town':
        
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
    
    #%% Load the nominal pressure data and prepare for training
    
    '''
    8.   P R E P A R E   T R A I N I N G   D A T A    
    '''
    
    if args.wdn == 'l-town':    
    
        print('Pre-processing nominal pressure data for training...\n')     
    
        # Populate feature vector x and label vector y from the nominal pressures
        # Also retrieve the scale and bias of the scaling transformation
        # This is so we can inverse transform the predicted values to calculate
        # relative reconstruction errors
        x,y,scale,bias = dataCleaner(pressure_df    = nominal_pressure, # Pass the nodal pressures
                                     observed_nodes = sensors,          # Indicate which nodes have sensors
                                     rescale        = args.scaling)     # Perform scaling on the timeseries data
        
        # Split the data into training and validation sets
        x_trn, x_val, y_trn, y_val = train_test_split(x, y, 
                                                      test_size    = 0.2,
                                                      random_state = 1,
                                                      shuffle      = False)
     
        
    #%% Load the nominal pressure data and prepare for training
    
    '''
    9.   S E T U P   F O R   T R A I N I N G 
    '''
    
    print('Setting up training session and creating model...\n')
    
    # ----------------
    # Hyper-parameters
    # ----------------
    batch_size    = 40
    learning_rate = 3e-4
    decay         = 6e-6
    shuffle       = False
    epochs        = args.epochs
    
    # --------------
    # Training setup
    # --------------
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate training and test set data generators
    trn_gnrtr = dataGenerator(G, x_trn, y_trn, batch_size, shuffle)
    val_gnrtr = dataGenerator(G, x_val, y_val, batch_size, shuffle)
    
    # Instantiate a Chebysev Network GNN model
    model     = ChebNet(name           = 'ChebNet',
                        data_generator = trn_gnrtr,
                        device         = device, 
                        in_channels    = np.shape(x_trn)[-1], 
                        out_channels   = np.shape(y_trn)[-1],
                        data_scale     = scale, 
                        data_bias      = bias).to(device)
    
    # If we've already train a model with the same setup
    if execution_no > 1:
        # We offer the user the option to load the previously trained weights
        if ui.yes_no_menu("A previous version of this model was found, do you want to load it ( 'yes' / 'no' ) ?\t"):
           model.load_model(last_model_path, last_log_path)
            # model.load_state_dict(torch.load(last_model_path))
    
    # Instantiate an optimizer
    optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=decay),
                                  dict(params=model.conv2.parameters(), weight_decay=decay),
                                  dict(params=model.conv3.parameters(), weight_decay=decay),
                                  dict(params=model.conv4.parameters(), weight_decay=0)],
                                  lr  = learning_rate,
                                  eps = 1e-7)
    
    # Configure an early stopping callback
    estop    = EarlyStopping(min_delta=.00001, patience=30)
    
    #%% Train the model
    
    '''
    10.   T R A I N 
    '''
        
    if ui.yes_no_menu("Do you want to train the model ( 'yes' / 'no' ) ?\t"):
        
        print("Training starting...\n")

        # Train for the predefined number of epochs
        for epoch in range(1, epochs+1):
            
            # Start a stopwatch timer
            start_time = time.time()
            
            # Train a single epoch, passing the optimizer and current epoch number
            model.train_one_epoch(optimizer = optimizer)
            
            # Validate the model after the gradient update
            model.validate()
            
            # Update the model results for the current epoch
            model.update_results()
            
            # Print stats for the epoch and the execution time
            model.print_stats(time.time() - start_time)
            
            # If this is the best model
            if model.val_loss < model.best_val_loss:
                # We save it
                torch.save(model.state_dict(), model_path)
            
            # If model is not improving
            if estop.step(torch.tensor(model.val_loss)):
                print('Early stopping activated...')
                break
        
        print("\nSaving training results to '{}'...\n".format(log_path))
        model.results.to_csv(log_path)

    
    # %% Predict a single unseen value
    '''
    11.   P R E D I C T   &   P L O T   A   S I N G L E   U N S E E N   S I G N A L
    '''
    
    rand_idx             = np.random.randint(0,200)                 # Pick a random signal
    
    n_nodes              = x_tst[rand_idx].shape[0]                 # Count number of nodes for later reshaping
    partial_graph_signal = x_tst[rand_idx]                          # Define the partial graph signal as a random signal from an unseen test set
    pred_graph_signal    = model.predict(G,partial_graph_signal)    # Predict a single partial signal given a graph G
    real_graph_signal    = y_tst[rand_idx].reshape(n_nodes,)        # Define the real signal, as a signal from the label test set with matching idx
        
    cmap      = plt.get_cmap('hot')             # Generate a colourmap
    part_cmap = cmap(partial_graph_signal[:,0]) # Fit the partial signal to the map...
    pred_cmap = cmap(pred_graph_signal)         # ... the predicted signal ...
    real_cmap = cmap(real_graph_signal)         # ... and the real signal.
    
    # Initalise a canvas to plot on
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,8), dpi=200, sharex=True, sharey=True) 
    
    # Visualise the the model using our visualisation utility
    ax[0] = visualise(G, pos=pos, color=part_cmap, figsize=figsize, edge_labels=True, axis=ax[0], cmap=part_cmap)
    ax[1] = visualise(G, pos=pos, color=pred_cmap, figsize=figsize, edge_labels=True, axis=ax[1])
    ax[2] = visualise(G, pos=pos, color=real_cmap, figsize=figsize, edge_labels=True, axis=ax[2])
    
    # Set titles of the subplots
    ax[0].set_title('Input',        fontsize='xx-large')
    ax[1].set_title('Prediction',   fontsize='xx-large')
    ax[2].set_title('True signal',  fontsize='xx-large')
    
    # Initialise a colorbar
    sm    = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    
    # Print a heading
    plt.suptitle("{} after {} epochs\n Sensors at nodes: {}, test signal number: {}".format(model.name, model.epoch,sensors,rand_idx), fontsize='xx-large')
    
    # Plot colorbar
    plt.colorbar(sm, ax=ax.ravel().tolist())
    
    # Display
    plt.show()