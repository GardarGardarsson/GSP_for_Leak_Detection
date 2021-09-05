#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:32:17 2021

@author: gardar
"""

import os
import yaml
import time
import torch
import epynet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.epanet_loader import get_nx_graph
from utils.epanet_simulator import epanetSimulator
from utils.data_loader import battledimLoader, dataCleaner, dataGenerator, embedSignalOnGraph, rescaleSignal
from modules.torch_gnn import ChebNet
from utils.visualisation import visualise


def predict_pressure(graph, pressure_series, print_out_rate=100, save=True, filename='reconstructions.csv'):
    
    results = []
    elapsed_time = time.time()
    
    for i, partial_graph_signal in enumerate(pressure_series):
        if not i % print_out_rate:
            execution_time = time.time()
        
        results.append(model.predict(graph, partial_graph_signal))
        
        if not i % print_out_rate:
            print('Signal:\t{}\t Execution:\t{:.3f} s\t Elapsed:\t{:.3f} s'.format(i,
                                                                                   time.time()-execution_time, 
                                                                                   time.time()-elapsed_time))
    if save:
        print('-'*63+"\nSaving results to: \n{}/{}\n\n".format(os.getcwd(),filename))
        pd.DataFrame(results).to_csv(filename)
    
    return results


if __name__ == '__main__':
    
    '''
     0. Configure Runtime
    '''
    print('+' + '-'*63 + '+')
    print('|' + ' '*63 + '|')
    print('|' + '{:^63s}'.format("R E C O N S T R U C T . P Y") + '|')
    print('|' + ' '*63 + '|')
    print('+' + '-'*63 + '+')
    print('\n')
    
    # Runtime configuration
    path_to_wdn     = './data/L-TOWN.inp'
    path_to_data    = './data/l-town-data/'
    weight_mode     = 'pipe_length'
    self_loops      = True
    scaling         = 'minmax'
    figsize         = (50,16)
    print_out_rate  = 100               
    model_name      = 'l-town-chebnet-' + weight_mode +'-' + scaling + '{}'.format('-self_loop' if self_loops else '')
    last_model_path = './studies/models/' + model_name + '-1.pt'
    last_log_path   = './studies/logs/'   + model_name + '-1.csv' 
        
    '''
     1. Make Graph
    '''
    print('1. Making graph from EPANET input file \n')
    
    # Import the .inp file using the EPYNET library
    wdn = epynet.Network(path_to_wdn)
    
    # Solve hydraulic model for a single timestep
    wdn.solve()
    
    # Convert the file using a custom function, based on:
    # https://github.com/BME-SmartLab/GraphConvWat 
    G , pos , head = get_nx_graph(wdn, weight_mode=weight_mode, get_head=True)
    
    # Open the dataset configuration file
    with open(path_to_data + 'dataset_configuration.yml') as file:
    
        # Load the configuration to a dictionary
        config = yaml.load(file, Loader=yaml.FullLoader) 
    
    # Generate a list of integers, indicating the number of the node
    # at which a  pressure sensor is present
    sensors = [int(string.replace("n", "")) for string in config['pressure_sensors']]
    
    if self_loops:
        for sensor_node in sensors:             # For each node in the sensor list
            G.add_edge(u_of_edge=sensor_node,   # Add an edge from that node ...
                       v_of_edge=sensor_node,   # ... to itself ...
                       weight=1.,name='SELF')   # ... and set its weight to equal 1
    
    '''
     2. Get Simulation Data
    '''
    print('2. Retrieving simulation data \n')
    
    # Instantiate the nominal WDN model
    nominal_wdn_model = epanetSimulator(path_to_wdn, path_to_data)
    
    # Run a simulation
    nominal_wdn_model.simulate()
    
    # Retrieve the nodal pressures
    nominal_pressure = nominal_wdn_model.get_simulated_pressure()
    
    x,y,scale,bias = dataCleaner(pressure_df    = nominal_pressure, # Pass the nodal pressures
                                 observed_nodes = sensors,          # Indicate which nodes have sensors
                                 rescale        = scaling)          # Perform scaling on the timeseries data
    
    # Split the data into training and validation sets
    x_trn, x_val, y_trn, y_val = train_test_split(x, y, 
                                                  test_size    = 0.2,
                                                  random_state = 1,
                                                  shuffle      = False)
    
    '''
     3. Get Historical Data
    '''
    print('3. Retrieving historical data \n')
    
    # Load the data into a numpy array with format matching the GraphConvWat problem
    pressure_2018 = battledimLoader(observed_nodes = sensors,
                                    n_nodes        = 782,
                                    path           = path_to_data,
                                    file           = '2018_SCADA_Pressures.csv',
                                    rescale        = True, 
                                    scale          = scale,
                                    bias           = bias)
    
    pressure_2019 = battledimLoader(observed_nodes = sensors,
                                    n_nodes        = 782,
                                    path           = path_to_data,
                                    file           = '2019_SCADA_Pressures.csv',
                                    rescale        = True, 
                                    scale          = scale,
                                    bias           = bias)
    
    '''
     4. Load a Trained GNN Model
    '''
    print('4. Loading trained GNN model \n')
    
    # Set the computation device as NVIDIA GPU if available else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate a Chebysev Network GNN model
    model  = ChebNet(name           = 'ChebNet',
                     data_generator = None,
                     device         = device, 
                     in_channels    = np.shape(x_trn)[-1], 
                     out_channels   = np.shape(y_trn)[-1],
                     data_scale     = scale, 
                     data_bias      = bias).to(device)
    
    # We offer the user the option to load the previously trained weights
    model.load_model(last_model_path, last_log_path)
    
    '''
     5. Predict a Year's worth of Data
    '''
    print('\n')
    print('5.1 Predicting 2018 pressure\n')
    
    prediction_2018 = predict_pressure(G, 
                                       pressure_2018,
                                       print_out_rate = print_out_rate, 
                                       save           = True, 
                                       filename       = '2018_reconstructions.csv')
    
    print('5.2 Predicting 2019 pressure\n')
    
    prediction_2019 = predict_pressure(G, 
                                       pressure_2019,
                                       print_out_rate = print_out_rate, 
                                       save           = True, 
                                       filename       = '2019_reconstructions.csv')
    
    print('+' + '-'*63 + '+')
    print('|' + ' '*63 + '|')
    print('|' + '{:^63s}'.format("E X E C U T I O N   C O M P L E T E D") + '|')
    print('|' + ' '*63 + '|')
    print('+' + '-'*63 + '+')
